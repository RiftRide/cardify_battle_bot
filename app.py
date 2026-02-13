import os
import io
import re
import uuid
import json
import sqlite3
import logging
import random
import base64
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from PIL import Image
import anthropic

# ---------- Config ----------
BOT_TOKEN = os.getenv("BOT_TOKEN")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PORT = int(os.getenv("PORT", 10000))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing in environment.")
if not RENDER_EXTERNAL_URL:
    raise RuntimeError("RENDER_EXTERNAL_URL missing in environment.")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY missing in environment.")

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = f"{RENDER_EXTERNAL_URL}{WEBHOOK_PATH}"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pfp-battle-bot")

# ---------- FastAPI ----------
app = FastAPI()

try:
    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    log.warning(f"Templates/static not found: {e}")

os.makedirs("battles", exist_ok=True)
os.makedirs("cards", exist_ok=True)

# ---------- SQLite storage ----------
DB_PATH = "battles.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS battles (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            challenger_username TEXT,
            challenger_stats TEXT,
            opponent_username TEXT,
            opponent_stats TEXT,
            winner TEXT,
            html_path TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()

# ---------- In-memory state ----------
pending_challenges: dict[int, str] = {}
uploaded_cards: dict[int, dict] = {}

# ---------- Claude Vision ----------
RARITY_BONUS = {
    "common": 0,
    "rare": 20,
    "ultrarare": 40,
    "ultra-rare": 40,
    "legendary": 60,
}

claude_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


async def analyze_card_with_claude(file_bytes: bytes) -> dict:
    """Use Claude Vision API to extract card stats - reads exact values from card"""
    try:
        base64_image = base64.standard_b64encode(file_bytes).decode("utf-8")

        image = Image.open(io.BytesIO(file_bytes))
        image_format = image.format.lower() if image.format else "jpeg"
        if image_format in ["jpeg", "jpg", "png", "gif", "webp"]:
            media_type = f"image/{image_format}"
        else:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            file_bytes = buf.getvalue()
            base64_image = base64.standard_b64encode(file_bytes).decode("utf-8")
            media_type = "image/png"

        message = await claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Extract stats from this PFP battle card.

Look for these stats printed on the card:
- Power (attack stat) - read the EXACT number shown on the card, can be any value
- Defense (defense stat) - read the EXACT number shown on the card, can be any value
- Rarity (usually Common, Rare, Ultra-Rare, or Legendary)
- Serial Number - usually labeled "Serial", "S/N", "#", or shown as a fraction like "422/1999". Use the first number before any slash. Can be any value.

Return ONLY valid JSON (no markdown, no code blocks):
{"power": <exact number from card>, "defense": <exact number from card>, "rarity": "<exact rarity from card>", "serial": <exact serial number from card>}

IMPORTANT: Read the EXACT values printed on the card. Do NOT cap or limit numbers.
If power shows 500, return 500. If serial shows 692444, return 692444.
Only use defaults if a stat is truly not visible: power=50, defense=50, rarity="Common", serial=1000"""
                        }
                    ],
                }
            ],
        )

        response_text = message.content[0].text.strip()
        log.info(f"Claude response: {response_text[:200]}")

        # Parse JSON
        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()

        stats = json.loads(json_text)

        # Read exact values, only prevent negatives
        power = max(1, int(stats.get("power", 50)))
        defense = max(1, int(stats.get("defense", 50)))
        rarity = str(stats.get("rarity", "Common"))
        serial = max(1, int(stats.get("serial", 1000)))

        log.info(f"Extracted: power={power}, defense={defense}, rarity={rarity}, serial={serial}")

        return {
            "power": power,
            "defense": defense,
            "rarity": rarity,
            "serial": serial,
        }

    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Claude JSON: {e}")
        return {"power": 50, "defense": 50, "rarity": "Common", "serial": 1000}
    except anthropic.APIError as e:
        log.error(f"Anthropic API error: {e}")
        return {"power": 50, "defense": 50, "rarity": "Common", "serial": 1000}
    except Exception as e:
        log.exception(f"Claude API error: {e}")
        return {"power": 50, "defense": 50, "rarity": "Common", "serial": 1000}


# ---------- HP calculation ----------
def calculate_hp(card: dict) -> int:
    base = card.get("power", 50) + card.get("defense", 50)

    # Normalize rarity key: "Ultra-Rare" -> "ultrarare", "Ultra Rare" -> "ultrarare"
    rarity_key = card.get("rarity", "Common").lower().replace(" ", "").replace("-", "")
    rarity_bonus = RARITY_BONUS.get(rarity_key, 0)

    serial = int(card.get("serial", 1000))
    # Lower serial = rarer = bonus. Scale works for any serial range.
    # Serial 1 gets max bonus, higher serials get less
    if serial < 2000:
        serial_bonus = (2000 - serial) / 50.0
    else:
        # For very high serials, give diminishing small bonus based on inverse
        serial_bonus = max(0, 10.0 - (serial / 10000.0))

    hp = int(base + rarity_bonus + serial_bonus)
    return max(1, hp)


# ---------- Battle simulation ----------
def simulate_battle(hp1: int, hp2: int, power1: int, power2: int):
    """Return (final_hp1, final_hp2, battle_log)"""
    battle_log = []
    round_num = 0

    # Scale damage to work with any power range
    # Battles should last roughly 8-20 rounds
    avg_hp = (hp1 + hp2) / 2.0
    avg_power = (power1 + power2) / 2.0
    if avg_power > 0:
        # Target ~12 rounds average
        damage_scale = avg_hp / (avg_power * 12.0)
        damage_scale = max(0.01, min(damage_scale, 10.0))
    else:
        damage_scale = 1.0

    while hp1 > 0 and hp2 > 0 and round_num < 100:
        round_num += 1
        dmg1 = max(1, int(power1 * damage_scale * random.uniform(0.6, 1.4)))
        dmg2 = max(1, int(power2 * damage_scale * random.uniform(0.6, 1.4)))

        hp2 -= dmg1
        battle_log.append({
            "round": round_num,
            "attacker": 1,
            "damage": dmg1,
            "hp1": max(0, hp1),
            "hp2": max(0, hp2),
        })

        if hp2 <= 0:
            break

        hp1 -= dmg2
        battle_log.append({
            "round": round_num,
            "attacker": 2,
            "damage": dmg2,
            "hp1": max(0, hp1),
            "hp2": max(0, hp2),
        })

    return max(0, hp1), max(0, hp2), battle_log


# ---------- Battle HTML ----------
def save_battle_html(battle_id: str, battle_context: dict):
    """Generate battle replay HTML."""
    os.makedirs("battles", exist_ok=True)

    c1 = battle_context
    log_html = ""
    for e in battle_context.get("battle_log", [])[:20]:
        attacker = c1["card1_name"] if e["attacker"] == 1 else c1["card2_name"]
        log_html += (
            f'<div>R{e["round"]}: @{attacker} \u2192 {e["damage"]} dmg '
            f'(HP: {e["hp1"]} vs {e["hp2"]})</div>\n'
        )

    html = f"""<!DOCTYPE html>
<html><head><title>Battle {battle_id}</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body{{background:#0a0a1e;color:#fff;font-family:Arial;padding:20px;text-align:center}}
.arena{{background:rgba(255,255,255,0.05);border-radius:15px;padding:20px;margin:20px auto;max-width:700px}}
.fighters{{display:flex;justify-content:space-around;margin:20px 0}}
.fighter{{flex:1;padding:10px}}
.name{{font-size:1.3em;color:#ffd93d;margin-bottom:10px}}
.stats{{background:rgba(0,0,0,0.3);padding:10px;border-radius:8px}}
.stat{{margin:5px 0;font-size:0.9em}}
.vs{{font-size:2.5em;color:#ff6b6b;margin:0 15px;align-self:center}}
.winner{{background:linear-gradient(135deg,#667eea,#764ba2);padding:15px;border-radius:10px;margin:15px 0;font-size:1.3em}}
.hp-section{{margin:15px 0}}
.hp-row{{margin:8px 0;text-align:left;max-width:500px;margin-left:auto;margin-right:auto}}
.hp-bar-bg{{width:100%;height:24px;background:rgba(0,0,0,0.5);border-radius:12px;overflow:hidden;margin-top:4px}}
.hp-bar{{height:100%;border-radius:12px;transition:width 2s ease-out}}
.hp-bar.green{{background:linear-gradient(90deg,#4CAF50,#8BC34A)}}
.hp-bar.red{{background:linear-gradient(90deg,#f44336,#ff5722)}}
.log{{background:rgba(0,0,0,0.3);padding:15px;border-radius:10px;max-height:250px;overflow-y:auto;text-align:left;margin-top:20px}}
.log div{{padding:5px;margin:3px 0;background:rgba(255,255,255,0.03);border-left:3px solid #ff6b6b;font-size:0.85em}}
</style></head><body>
<h1>\u2694\ufe0f Battle Replay</h1>
<div class="arena">
<div class="fighters">
<div class="fighter">
<div class="name">@{c1['card1_name']}</div>
<div class="stats">
<div class="stat">\u26a1 Power: {c1['card1_stats']['power']}</div>
<div class="stat">\U0001f6e1 Defense: {c1['card1_stats']['defense']}</div>
<div class="stat">\u2728 {c1['card1_stats']['rarity']}</div>
<div class="stat">\U0001f3ab #{c1['card1_stats']['serial']}</div>
</div></div>
<div class="vs">VS</div>
<div class="fighter">
<div class="name">@{c1['card2_name']}</div>
<div class="stats">
<div class="stat">\u26a1 Power: {c1['card2_stats']['power']}</div>
<div class="stat">\U0001f6e1 Defense: {c1['card2_stats']['defense']}</div>
<div class="stat">\u2728 {c1['card2_stats']['rarity']}</div>
<div class="stat">\U0001f3ab #{c1['card2_stats']['serial']}</div>
</div></div></div>
<div class="winner">{'\U0001f3c6 Winner: @' + c1['winner_name'] if c1['winner_name'] != 'Tie' else '\U0001f91d Tie!'}</div>
<div class="hp-section">
<div class="hp-row">
<span>@{c1['card1_name']}: {c1['hp1_end']}/{c1['hp1_start']} HP</span>
<div class="hp-bar-bg"><div class="hp-bar {'green' if c1['hp1_end'] > 0 else 'red'}" id="hp1" style="width:100%"></div></div>
</div>
<div class="hp-row">
<span>@{c1['card2_name']}: {c1['hp2_end']}/{c1['hp2_start']} HP</span>
<div class="hp-bar-bg"><div class="hp-bar {'green' if c1['hp2_end'] > 0 else 'red'}" id="hp2" style="width:100%"></div></div>
</div>
</div>
<div class="log"><h3>\U0001f4dc Battle Log</h3>{log_html}</div>
</div>
<script>
setTimeout(()=>{{
document.getElementById('hp1').style.width='{max(0, int(c1["hp1_end"]/max(1,c1["hp1_start"])*100))}%';
document.getElementById('hp2').style.width='{max(0, int(c1["hp2_end"]/max(1,c1["hp2_start"])*100))}%';
}},500);
</script>
</body></html>"""

    path = f"battles/{battle_id}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def persist_battle_record(
    battle_id: str,
    challenger_username: str,
    challenger_stats: dict,
    opponent_username: str,
    opponent_stats: dict,
    winner: Optional[str],
    html_path: str,
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO battles (id, timestamp, challenger_username, challenger_stats, "
        "opponent_username, opponent_stats, winner, html_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            battle_id,
            datetime.utcnow().isoformat(),
            challenger_username,
            json.dumps(challenger_stats),
            opponent_username,
            json.dumps(opponent_stats),
            winner or "",
            html_path,
        ),
    )
    conn.commit()
    conn.close()


# ---------- Telegram handlers ----------
async def cmd_battle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\u2694\ufe0f PFP Battle Bot\n\n"
        "/challenge @username - Start a battle\n"
        "/mystats - View your card\n\n"
        "\U0001f916 Powered by Claude AI"
    )


async def cmd_challenge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].startswith("@"):
        await update.message.reply_text("Usage: /challenge @username")
        return

    challenger = update.effective_user
    opponent_username = context.args[0].lstrip("@").strip()

    if challenger.username and challenger.username.lower() == opponent_username.lower():
        await update.message.reply_text("\u274c You can't challenge yourself!")
        return

    pending_challenges[challenger.id] = opponent_username
    log.info(f"Challenge: @{challenger.username} -> @{opponent_username}")

    await update.message.reply_text(
        f"\u2694\ufe0f @{challenger.username} challenged @{opponent_username}!\n\n"
        "\U0001f4e4 Both players: upload your battle card image."
    )


async def cmd_mystats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    card = uploaded_cards.get(update.effective_user.id)
    if not card:
        await update.message.reply_text("\u274c Upload a card first!")
        return

    hp = calculate_hp(card)
    await update.message.reply_text(
        f"\U0001f4ca Your Card:\n"
        f"\u26a1 Power: {card['power']}\n"
        f"\U0001f6e1 Defense: {card['defense']}\n"
        f"\u2728 {card['rarity']}\n"
        f"\U0001f3ab #{card['serial']}\n"
        f"\u2764\ufe0f HP: {hp}"
    )


async def debug_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug handler to see what we're receiving"""
    if not update.message:
        return
    log.info(f"DEBUG: Message from {update.effective_user.username}")
    log.info(f"DEBUG: photo={bool(update.message.photo)} doc={bool(update.message.document)} text={bool(update.message.text)}")


async def handler_card_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    log.info(f"Card upload handler triggered! User: {update.effective_user.username}")

    user = update.effective_user
    username = (user.username or f"user{user.id}").lower()
    user_id = user.id

    try:
        # Get file
        file_obj = None
        if update.message.photo:
            log.info("Photo detected")
            file_obj = await update.message.photo[-1].get_file()
        elif update.message.document:
            log.info(f"Document detected: {update.message.document.mime_type}")
            file_obj = await update.message.document.get_file()
        else:
            log.warning("No photo or document found")
            return

        file_bytes = await file_obj.download_as_bytearray()

        if len(file_bytes) == 0:
            await update.message.reply_text("\u26a0\ufe0f Empty file received. Try again.")
            return

        log.info(f"Downloaded {len(file_bytes)} bytes")

        # Save
        save_path = f"cards/{username}.png"
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        msg = await update.message.reply_text("\U0001f916 Analyzing card...")

        # Await the async Claude call
        parsed = await analyze_card_with_claude(bytes(file_bytes))

        card = {
            "username": username,
            "user_id": user_id,
            "path": save_path,
            "power": int(parsed["power"]),
            "defense": int(parsed["defense"]),
            "rarity": parsed["rarity"],
            "serial": int(parsed["serial"]),
        }

        uploaded_cards[user_id] = card
        hp = calculate_hp(card)

        await msg.edit_text(
            f"\u2705 @{username} ready!\n"
            f"\u26a1 Power: {card['power']}\n"
            f"\U0001f6e1 Defense: {card['defense']}\n"
            f"\u2728 {card['rarity']}\n"
            f"\U0001f3ab Serial: #{card['serial']}\n"
            f"\u2764\ufe0f HP: {hp}"
        )

        # Battle trigger logic
        triggered_pair = None

        if user_id in pending_challenges:
            opp = pending_challenges[user_id].lower()
            opp_id = next(
                (uid for uid, c in uploaded_cards.items() if c["username"].lower() == opp),
                None,
            )
            if opp_id:
                triggered_pair = (user_id, opp_id)

        if not triggered_pair:
            for cid, opp in pending_challenges.items():
                if username == opp.lower() and cid in uploaded_cards:
                    triggered_pair = (cid, user_id)
                    break

        # Run battle
        if triggered_pair:
            cid, oid = triggered_pair
            c1, c2 = uploaded_cards[cid], uploaded_cards[oid]

            hp1_start, hp2_start = calculate_hp(c1), calculate_hp(c2)
            hp1_end, hp2_end, log_data = simulate_battle(
                hp1_start, hp2_start, c1["power"], c2["power"]
            )

            if hp1_end > hp2_end:
                winner = c1["username"]
            elif hp2_end > hp1_end:
                winner = c2["username"]
            else:
                winner = None

            bid = str(uuid.uuid4())
            ctx = {
                "card1_name": c1["username"],
                "card2_name": c2["username"],
                "card1_stats": {
                    "power": c1["power"],
                    "defense": c1["defense"],
                    "rarity": c1["rarity"],
                    "serial": c1["serial"],
                },
                "card2_stats": {
                    "power": c2["power"],
                    "defense": c2["defense"],
                    "rarity": c2["rarity"],
                    "serial": c2["serial"],
                },
                "hp1_start": hp1_start,
                "hp2_start": hp2_start,
                "hp1_end": hp1_end,
                "hp2_end": hp2_end,
                "winner_name": winner or "Tie",
                "battle_id": bid,
                "battle_log": log_data,
            }

            html_path = save_battle_html(bid, ctx)
            persist_battle_record(
                bid,
                c1["username"],
                ctx["card1_stats"],
                c2["username"],
                ctx["card2_stats"],
                winner,
                html_path,
            )

            url = f"{RENDER_EXTERNAL_URL}/battle/{bid}"
            kb = InlineKeyboardMarkup(
                [[InlineKeyboardButton("\U0001f3ac View Replay", url=url)]]
            )

            result = f"\u2694\ufe0f Battle Complete!\n\n"
            if winner:
                result += f"\U0001f3c6 @{winner}!\n\n"
            else:
                result += "\U0001f91d Tie!\n\n"
            result += (
                f"@{c1['username']}: {hp1_end}/{hp1_start} HP\n"
                f"@{c2['username']}: {hp2_end}/{hp2_start} HP\n\n"
                f"Battle lasted {len(set(e['round'] for e in log_data))} rounds!"
            )

            await update.message.reply_text(result, reply_markup=kb)

            uploaded_cards.pop(cid, None)
            uploaded_cards.pop(oid, None)
            pending_challenges.pop(cid, None)

        else:
            waiting = None
            if user_id in pending_challenges:
                waiting = f"@{pending_challenges[user_id]}"
            else:
                for cid, opp in pending_challenges.items():
                    if username == opp.lower():
                        cc = uploaded_cards.get(cid)
                        waiting = f"@{cc['username']}" if cc else "your challenger"
                        break

            if waiting:
                await update.message.reply_text(f"\u23f3 Waiting for {waiting}...")
            else:
                await update.message.reply_text(
                    "\u2705 Card uploaded! Use /challenge @username to battle!"
                )

    except Exception as e:
        log.exception(f"Card upload error: {e}")
        try:
            await update.message.reply_text(
                "\u274c Error processing card. Try again."
            )
        except:
            pass


# ---------- FastAPI routes ----------
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ok", "bot": "PFP Battle", "vision": "Claude API"}


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"healthy": True}


@app.get("/battle/{battle_id}")
async def battle_page(battle_id: str):
    path = f"battles/{battle_id}.html"
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return HTMLResponse("<h1>Battle Not Found</h1>", status_code=404)


@app.post(WEBHOOK_PATH)
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return JSONResponse({"ok": True})


# ---------- Startup / Shutdown ----------
telegram_app: Optional[Application] = None


@app.on_event("startup")
async def on_startup():
    global telegram_app
    log.info("Starting bot with Claude Vision...")

    telegram_app = Application.builder().token(BOT_TOKEN).build()
    telegram_app.add_handler(CommandHandler("battle", cmd_battle))
    telegram_app.add_handler(CommandHandler("start", cmd_battle))
    telegram_app.add_handler(CommandHandler("challenge", cmd_challenge))
    telegram_app.add_handler(CommandHandler("mystats", cmd_mystats))
    telegram_app.add_handler(MessageHandler(filters.PHOTO, handler_card_upload))
    telegram_app.add_handler(
        MessageHandler(filters.Document.IMAGE, handler_card_upload)
    )
    telegram_app.add_handler(MessageHandler(filters.ALL, debug_handler))

    await telegram_app.initialize()
    await telegram_app.bot.delete_webhook(drop_pending_updates=True)
    await telegram_app.bot.set_webhook(WEBHOOK_URL)
    log.info(f"Webhook set: {WEBHOOK_URL}")


@app.on_event("shutdown")
async def on_shutdown():
    if telegram_app:
        try:
            await telegram_app.bot.delete_webhook()
            await telegram_app.shutdown()
        except:
            pass
    log.info("Bot stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
