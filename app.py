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
RARITY_BONUS = {"common": 0, "rare": 20, "ultrarare": 40, "ultra-rare": 40, "legendary": 60}

# ⭐ FIX: Use AsyncAnthropic instead of Anthropic
claude_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# ⭐ FIX: Make this function async
async def analyze_card_with_claude(file_bytes: bytes) -> dict:
    """Use Claude Vision API to extract card stats - ASYNC version"""
    try:
        base64_image = base64.standard_b64encode(file_bytes).decode("utf-8")
        
        image = Image.open(io.BytesIO(file_bytes))
        image_format = image.format.lower() if image.format else "jpeg"
        media_type = f"image/{image_format}" if image_format in ["jpeg", "png", "gif", "webp"] else "image/jpeg"
        
        # ⭐ FIX: Await the async API call
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

Return ONLY valid JSON (no markdown):
{"power": <number 1-200>, "defense": <number 1-200>, "rarity": "Common|Rare|Ultra-Rare|Legendary", "serial": <number 1-1999>}

Defaults if unclear: power=50, defense=50, rarity="Common", serial=1000"""
                        }
                    ],
                }
            ],
        )
        
        response_text = message.content[0].text.strip()
        log.info(f"Claude response: {response_text[:150]}")
        
        # Parse JSON
        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()

        stats = json.loads(json_text)

        power = max(1, min(int(stats.get("power", 50)), 200))
        defense = max(1, min(int(stats.get("defense", 50)), 200))
        rarity = stats.get("rarity", "Common")
        serial = max(1, min(int(stats.get("serial", 1000)), 1999))

        log.info(f"Extracted: power={power}, defense={defense}, rarity={rarity}, serial={serial}")

        return {
            "power": power,
            "defense": defense,
            "rarity": rarity,
            "serial": serial
        }
        
    except Exception as e:
        log.exception(f"Claude API error: {e}")
        return {
            "power": 50,
            "defense": 50,
            "rarity": "Common",
            "serial": 1000
        }

# ---------- HP calculation ----------
def calculate_hp(card: dict) -> int:
    base = card.get("power", 50) + card.get("defense", 50)
    rarity_key = card.get("rarity", "Common").lower()
    rarity_bonus = RARITY_BONUS.get(rarity_key, 0)
    serial = int(card.get("serial", 1000))
    serial_bonus = (2000 - serial) / 50.0
    hp = int(base + rarity_bonus + serial_bonus)
    return max(1, hp)

# ---------- Battle simulation ----------
def simulate_battle(hp1: int, hp2: int, power1: int, power2: int):
    """Return (final_hp1, final_hp2, battle_log)"""
    battle_log = []
    round_num = 0
    
    while hp1 > 0 and hp2 > 0 and round_num < 100:
        round_num += 1
        dmg1 = max(1, int(power1 * random.uniform(0.08, 0.16)))
        dmg2 = max(1, int(power2 * random.uniform(0.08, 0.16)))
        
        hp2 -= dmg1
        battle_log.append({
            "round": round_num,
            "attacker": 1,
            "damage": dmg1,
            "hp1": max(0, hp1),
            "hp2": max(0, hp2)
        })
        
        if hp2 <= 0:
            break
        
        hp1 -= dmg2
        battle_log.append({
            "round": round_num,
            "attacker": 2,
            "damage": dmg2,
            "hp1": max(0, hp1),
            "hp2": max(0, hp2)
        })
    
    return max(0, hp1), max(0, hp2), battle_log

# ---------- Battle HTML (SIMPLIFIED) ----------
def save_battle_html(battle_id: str, battle_context: dict):
    """Generate battle replay HTML."""
    os.makedirs("battles", exist_ok=True)
    
    c1 = battle_context
    log_html = ""
    for e in battle_context.get("battle_log", [])[:15]:
        attacker = c1["card1_name"] if e["attacker"] == 1 else c1["card2_name"]
        log_html += f'<div>R{e["round"]}: @{attacker} → {e["damage"]} dmg (HP: {e["hp1"]} vs {e["hp2"]})</div>\n'
    
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
.vs{{font-size:2.5em;color:#ff6b6b;margin:0 15px}}
.winner{{background:linear-gradient(135deg,#667eea,#764ba2);padding:15px;border-radius:10px;margin:15px 0;font-size:1.3em}}
.log{{background:rgba(0,0,0,0.3);padding:15px;border-radius:10px;max-height:250px;overflow-y:auto;text-align:left}}
.log div{{padding:5px;margin:3px 0;background:rgba(255,255,255,0.03);border-left:3px solid #ff6b6b}}
</style></head><body>
<h1>⚔️ Battle Replay</h1>
<div class="arena">
<div class="fighters">
<div class="fighter">
<div class="name">@{c1['card1_name']}</div>
<div class="stats">
<div class="stat">⚡ Power: {c1['card1_stats']['power']}</div>
<div class="stat">🛡️ Defense: {c1['card1_stats']['defense']}</div>
<div class="stat">✨ {c1['card1_stats']['rarity']}</div>
<div class="stat">🎫 #{c1['card1_stats']['serial']}</div>
</div></div>
<div class="vs">VS</div>
<div class="fighter">
<div class="name">@{c1['card2_name']}</div>
<div class="stats">
<div class="stat">⚡ Power: {c1['card2_stats']['power']}</div>
<div class="stat">🛡️ Defense: {c1['card2_stats']['defense']}</div>
<div class="stat">✨ {c1['card2_stats']['rarity']}</div>
<div class="stat">🎫 #{c1['card2_stats']['serial']}</div>
</div></div></div>
<div class="winner">{'🏆 Winner: @' + c1['winner_name'] if c1['winner_name'] != 'Tie' else '🤝 Tie!'}</div>
<div style="margin:15px 0">
<div>@{c1['card1_name']}: {c1['hp1_end']}/{c1['hp1_start']} HP</div>
<div>@{c1['card2_name']}: {c1['hp2_end']}/{c1['hp2_start']} HP</div>
</div>
<div class="log"><h3>📜 Battle Log</h3>{log_html}</div>
</div></body></html>"""
    
    path = f"battles/{battle_id}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path

def persist_battle_record(battle_id, c_user, c_stats, o_user, o_stats, winner, html_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO battles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (battle_id, datetime.utcnow().isoformat(), c_user, json.dumps(c_stats),
         o_user, json.dumps(o_stats), winner or "", html_path)
    )
    conn.commit()
    conn.close()

# ---------- Telegram handlers ----------
async def cmd_battle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚔️ PFP Battle Bot\n\n"
        "/challenge @username - Start a battle\n"
        "/mystats - View your card\n\n"
        "🤖 Powered by Claude AI"
    )

async def cmd_challenge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].startswith("@"):
        await update.message.reply_text("Usage: /challenge @username")
        return
    
    challenger = update.effective_user
    opponent_username = context.args[0].lstrip("@").strip()
    
    if challenger.username and challenger.username.lower() == opponent_username.lower():
        await update.message.reply_text("❌ You can't challenge yourself!")
        return
    
    pending_challenges[challenger.id] = opponent_username
    log.info(f"Challenge: @{challenger.username} -> @{opponent_username}")
    
    await update.message.reply_text(
        f"⚔️ @{challenger.username} challenged @{opponent_username}!\n\n"
        "📤 Both players: upload your battle card image."
    )

async def cmd_mystats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    card = uploaded_cards.get(update.effective_user.id)
    if not card:
        await update.message.reply_text("❌ Upload a card first!")
        return
    
    hp = calculate_hp(card)
    await update.message.reply_text(
        f"📊 Your Card:\n"
        f"⚡ Power: {card['power']}\n"
        f"🛡️ Defense: {card['defense']}\n"
        f"✨ {card['rarity']}\n"
        f"🎫 #{card['serial']}\n"
        f"❤️ HP: {hp}"
    )

async def handler_card_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    username = (user.username or f"user{user.id}").lower()
    user_id = user.id

    try:
        # Get file
        file_obj = None
        if update.message.photo:
            file_obj = await update.message.photo[-1].get_file()
        elif update.message.document:
            file_obj = await update.message.document.get_file()
        else:
            return

        file_bytes = await file_obj.download_as_bytearray()
        
        # Save
        save_path = f"cards/{username}.png"
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        msg = await update.message.reply_text("🤖 Analyzing card...")

        # ⭐ FIX: AWAIT the async function
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
            f"✅ @{username} ready!\n"
            f"⚡{card['power']} 🛡️{card['defense']} ✨{card['rarity']} 🎫#{card['serial']}\n"
            f"❤️ HP: {hp}"
        )
    
        # Battle trigger logic
        triggered_pair = None

        if user_id in pending_challenges:
            opp = pending_challenges[user_id].lower()
            opp_id = next((uid for uid, c in uploaded_cards.items() if c["username"].lower() == opp), None)
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
            hp1_end, hp2_end, log_data = simulate_battle(hp1_start, hp2_start, c1["power"], c2["power"])

            winner = c1["username"] if hp1_end > hp2_end else (c2["username"] if hp2_end > hp1_end else None)

            bid = str(uuid.uuid4())
            ctx = {
                "card1_name": c1["username"], "card2_name": c2["username"],
                "card1_stats": {"power": c1["power"], "defense": c1["defense"], "rarity": c1["rarity"], "serial": c1["serial"]},
                "card2_stats": {"power": c2["power"], "defense": c2["defense"], "rarity": c2["rarity"], "serial": c2["serial"]},
                "hp1_start": hp1_start, "hp2_start": hp2_start,
                "hp1_end": hp1_end, "hp2_end": hp2_end,
                "winner_name": winner or "Tie", "battle_id": bid, "battle_log": log_data
            }

            html_path = save_battle_html(bid, ctx)
            persist_battle_record(bid, c1["username"], ctx["card1_stats"], c2["username"], ctx["card2_stats"], winner, html_path)

            url = f"{RENDER_EXTERNAL_URL}/battle/{bid}"
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🎬 View Replay", url=url)]])

            result = f"⚔️ Battle Complete!\n\n"
            result += f"🏆 @{winner}!\n\n" if winner else "🤝 Tie!\n\n"
            result += f"@{c1['username']}: {hp1_end}/{hp1_start} HP\n@{c2['username']}: {hp2_end}/{hp2_start} HP"

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
                await update.message.reply_text(f"⏳ Waiting for {waiting}...")
            else:
                await update.message.reply_text("✅ Use /challenge @username to battle!")

    except Exception as e:
        log.exception(f"Card upload error: {e}")
        try:
            await update.message.reply_text("❌ Error processing card. Try again.")
        except:
            pass

# ---------- FastAPI routes ----------
@app.get("/")
async def root():
    return {"status": "ok", "bot": "PFP Battle", "vision": "Claude API"}

@app.get("/health")
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

# ---------- Startup ----------
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
    telegram_app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, handler_card_upload))
    
    await telegram_app.initialize()

       # Set webhook after delay
       async def delayed_webhook():
          await asyncio.sleep(3)
          await telegram_app.bot.delete_webhook(drop_pending_updates=True)
          await telegram_app.bot.set_webhook(WEBHOOK_URL)
          log.info(f"Webhook: {WEBHOOK_URL}")
    
    webhook_task = asyncio.create_task(delayed_webhook())
    log.info("Bot ready")

@app.on_event("shutdown")
async def on_shutdown():
    if telegram_app:
        try:
            await telegram_app.bot.delete_webhook()
            await telegram_app.shutdown()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
    
