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

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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
    raise RuntimeError("BOT_TOKEN missing")
if not RENDER_EXTERNAL_URL:
    raise RuntimeError("RENDER_EXTERNAL_URL missing")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY missing")

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = f"{RENDER_EXTERNAL_URL}{WEBHOOK_PATH}"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pfp-battle-bot")

# ---------- FastAPI ----------
app = FastAPI()

os.makedirs("battles", exist_ok=True)
os.makedirs("cards", exist_ok=True)

# ---------- Database ----------
DB_PATH = "battles.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS battles (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        challenger_username TEXT,
        challenger_stats TEXT,
        opponent_username TEXT,
        opponent_stats TEXT,
        winner TEXT,
        html_path TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

# ---------- State ----------
pending_challenges: dict[int, str] = {}
uploaded_cards: dict[int, dict] = {}

# ---------- Claude Vision ----------
RARITY_BONUS = {"common": 0, "rare": 20, "ultrarare": 40, "ultra-rare": 40, "legendary": 60}

claude_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

async def analyze_card_with_claude(file_bytes: bytes) -> dict:
    """Extract stats using Claude Vision API"""
    try:
        base64_image = base64.standard_b64encode(file_bytes).decode("utf-8")
        image = Image.open(io.BytesIO(file_bytes))
        fmt = (image.format or "jpeg").lower()
        media_type = f"image/{fmt}" if fmt in ["jpeg", "png", "gif", "webp"] else "image/jpeg"
        
        message = await claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_image}},
                    {"type": "text", "text": 'Extract stats from this PFP battle card. Return ONLY JSON: {"power": <1-200>, "defense": <1-200>, "rarity": "Common|Rare|Ultra-Rare|Legendary", "serial": <1-1999>}. Defaults: power=50, defense=50, rarity="Common", serial=1000'}
                ]
            }]
        )
        
        text = message.content[0].text.strip()
        log.info(f"Claude: {text[:150]}")
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        stats = json.loads(text)
        return {
            "power": max(1, min(int(stats.get("power", 50)), 200)),
            "defense": max(1, min(int(stats.get("defense", 50)), 200)),
            "rarity": stats.get("rarity", "Common"),
            "serial": max(1, min(int(stats.get("serial", 1000)), 1999))
        }
    except Exception as e:
        log.exception(f"Claude error: {e}")
        return {"power": 50, "defense": 50, "rarity": "Common", "serial": 1000}

def calculate_hp(card: dict) -> int:
    base = card.get("power", 50) + card.get("defense", 50)
    rarity_bonus = RARITY_BONUS.get(card.get("rarity", "Common").lower(), 0)
    serial_bonus = (2000 - int(card.get("serial", 1000))) / 50.0
    return max(1, int(base + rarity_bonus + serial_bonus))

def simulate_battle(hp1: int, hp2: int, power1: int, power2: int):
    log_data, round_num = [], 0
    while hp1 > 0 and hp2 > 0 and round_num < 100:
        round_num += 1
        dmg1 = max(1, int(power1 * random.uniform(0.08, 0.16)))
        dmg2 = max(1, int(power2 * random.uniform(0.08, 0.16)))
        hp2 -= dmg1
        log_data.append({"round": round_num, "attacker": 1, "damage": dmg1, "hp1": max(0, hp1), "hp2": max(0, hp2)})
        if hp2 <= 0:
            break
        hp1 -= dmg2
        log_data.append({"round": round_num, "attacker": 2, "damage": dmg2, "hp1": max(0, hp1), "hp2": max(0, hp2)})
    return max(0, hp1), max(0, hp2), log_data

def save_battle_html(battle_id: str, ctx: dict):
    os.makedirs("battles", exist_ok=True)
    log_html = "".join(f'<div>R{e["round"]}: @{ctx["card1_name"] if e["attacker"]==1 else ctx["card2_name"]} → {e["damage"]} dmg</div>' for e in ctx.get("battle_log", [])[:15])
    html = f'''<!DOCTYPE html><html><head><title>Battle {battle_id}</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{{background:#0a0a1e;color:#fff;font-family:Arial;padding:20px;text-align:center}}.arena{{background:rgba(255,255,255,0.05);border-radius:15px;padding:20px;margin:20px auto;max-width:700px}}.fighters{{display:flex;justify-content:space-around;margin:20px 0}}.fighter{{flex:1;padding:10px}}.name{{font-size:1.3em;color:#ffd93d;margin-bottom:10px}}.stats{{background:rgba(0,0,0,0.3);padding:10px;border-radius:8px}}.stat{{margin:5px 0;font-size:0.9em}}.vs{{font-size:2.5em;color:#ff6b6b;margin:0 15px}}.winner{{background:linear-gradient(135deg,#667eea,#764ba2);padding:15px;border-radius:10px;margin:15px 0;font-size:1.3em}}.log{{background:rgba(0,0,0,0.3);padding:15px;border-radius:10px;max-height:250px;overflow-y:auto;text-align:left}}.log div{{padding:5px;margin:3px 0;background:rgba(255,255,255,0.03);border-left:3px solid #ff6b6b}}</style></head><body><h1>⚔️ Battle Replay</h1><div class="arena"><div class="fighters"><div class="fighter"><div class="name">@{ctx['card1_name']}</div><div class="stats"><div class="stat">⚡ {ctx['card1_stats']['power']}</div><div class="stat">🛡️ {ctx['card1_stats']['defense']}</div><div class="stat">✨ {ctx['card1_stats']['rarity']}</div><div class="stat">🎫 #{ctx['card1_stats']['serial']}</div></div></div><div class="vs">VS</div><div class="fighter"><div class="name">@{ctx['card2_name']}</div><div class="stats"><div class="stat">⚡ {ctx['card2_stats']['power']}</div><div class="stat">🛡️ {ctx['card2_stats']['defense']}</div><div class="stat">✨ {ctx['card2_stats']['rarity']}</div><div class="stat">🎫 #{ctx['card2_stats']['serial']}</div></div></div></div><div class="winner">{'🏆 @' + ctx['winner_name'] if ctx['winner_name'] != 'Tie' else '🤝 Tie!'}</div><div style="margin:15px 0"><div>@{ctx['card1_name']}: {ctx['hp1_end']}/{ctx['hp1_start']} HP</div><div>@{ctx['card2_name']}: {ctx['hp2_end']}/{ctx['hp2_start']} HP</div></div><div class="log"><h3>📜 Battle Log</h3>{log_html}</div></div></body></html>'''
    path = f"battles/{battle_id}.html"
    with open(path, "w") as f:
        f.write(html)
    return path

def persist_battle(bid, c_user, c_stats, o_user, o_stats, winner, path):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO battles VALUES (?,?,?,?,?,?,?,?)", (bid, datetime.utcnow().isoformat(), c_user, json.dumps(c_stats), o_user, json.dumps(o_stats), winner or "", path))
    conn.commit()
    conn.close()

# ---------- Telegram Handlers ----------
async def cmd_battle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚔️ PFP Battle Bot\n\n/challenge @user\n/mystats\n\n🤖 Claude AI")

async def cmd_challenge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].startswith("@"):
        await update.message.reply_text("Usage: /challenge @username")
        return
    
    opp = context.args[0].lstrip("@").strip()
    if update.effective_user.username and update.effective_user.username.lower() == opp.lower():
        await update.message.reply_text("❌ Can't challenge yourself!")
        return
    
    pending_challenges[update.effective_user.id] = opp
    log.info(f"Challenge: @{update.effective_user.username} -> @{opp}")
    await update.message.reply_text(f"⚔️ @{update.effective_user.username} challenged @{opp}!\n📤 Both upload cards.")

async def cmd_mystats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    card = uploaded_cards.get(update.effective_user.id)
    if not card:
        await update.message.reply_text("❌ Upload a card first!")
        return
    hp = calculate_hp(card)
    await update.message.reply_text(f"📊 Card:\n⚡{card['power']} 🛡️{card['defense']}\n✨{card['rarity']} 🎫#{card['serial']}\n❤️{hp} HP")

async def handler_card_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    username = (user.username or f"user{user.id}").lower()
    
    try:
        file_obj = None
        if update.message.photo:
            file_obj = await update.message.photo[-1].get_file()
        elif update.message.document:
            file_obj = await update.message.document.get_file()
        else:
            return

        file_bytes = await file_obj.download_as_bytearray()
        with open(f"cards/{username}.png", "wb") as f:
            f.write(file_bytes)

        msg = await update.message.reply_text("🤖 Analyzing...")
        parsed = await analyze_card_with_claude(bytes(file_bytes))

        card = {
            "username": username, "user_id": user.id, "path": f"cards/{username}.png",
            "power": parsed["power"], "defense": parsed["defense"],
            "rarity": parsed["rarity"], "serial": parsed["serial"]
        }
        uploaded_cards[user.id] = card
        hp = calculate_hp(card)

        await msg.edit_text(f"✅ @{username}\n⚡{card['power']} 🛡️{card['defense']} ✨{card['rarity']} 🎫#{card['serial']}\n❤️{hp} HP")
    
        # Battle trigger
        triggered = None
        if user.id in pending_challenges:
            opp = pending_challenges[user.id].lower()
            opp_id = next((uid for uid, c in uploaded_cards.items() if c["username"].lower() == opp), None)
            if opp_id:
                triggered = (user.id, opp_id)

        if not triggered:
            for cid, opp in pending_challenges.items():
                if username == opp.lower() and cid in uploaded_cards:
                    triggered = (cid, user.id)
                    break

        if triggered:
            cid, oid = triggered
            c1, c2 = uploaded_cards[cid], uploaded_cards[oid]
            hp1_s, hp2_s = calculate_hp(c1), calculate_hp(c2)
            hp1_e, hp2_e, log_d = simulate_battle(hp1_s, hp2_s, c1["power"], c2["power"])
            winner = c1["username"] if hp1_e > hp2_e else (c2["username"] if hp2_e > hp1_e else None)

            bid = str(uuid.uuid4())
            ctx = {
                "card1_name": c1["username"], "card2_name": c2["username"],
                "card1_stats": {"power": c1["power"], "defense": c1["defense"], "rarity": c1["rarity"], "serial": c1["serial"]},
                "card2_stats": {"power": c2["power"], "defense": c2["defense"], "rarity": c2["rarity"], "serial": c2["serial"]},
                "hp1_start": hp1_s, "hp2_start": hp2_s, "hp1_end": hp1_e, "hp2_end": hp2_e,
                "winner_name": winner or "Tie", "battle_id": bid, "battle_log": log_d
            }

            path = save_battle_html(bid, ctx)
            persist_battle(bid, c1["username"], ctx["card1_stats"], c2["username"], ctx["card2_stats"], winner, path)

            url = f"{RENDER_EXTERNAL_URL}/battle/{bid}"
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🎬 Replay", url=url)]])
            result = f"⚔️ Battle!\n\n{'🏆 @' + winner if winner else '🤝 Tie'}\n\n@{c1['username']}: {hp1_e}/{hp1_s}\n@{c2['username']}: {hp2_e}/{hp2_s}"
            await update.message.reply_text(result, reply_markup=kb)

            uploaded_cards.pop(cid, None)
            uploaded_cards.pop(oid, None)
            pending_challenges.pop(cid, None)
        else:
            waiting = None
            if user.id in pending_challenges:
                waiting = f"@{pending_challenges[user.id]}"
            else:
                for cid, opp in pending_challenges.items():
                    if username == opp.lower():
                        cc = uploaded_cards.get(cid)
                        waiting = f"@{cc['username']}" if cc else "challenger"
                        break
            await update.message.reply_text(f"⏳ Waiting for {waiting}..." if waiting else "✅ Use /challenge!")

    except Exception as e:
        log.exception(f"Upload error: {e}")
        try:
            await update.message.reply_text("❌ Error. Try again.")
        except:
            pass

# ---------- Routes ----------
@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"healthy": True}

@app.get("/battle/{bid}")
async def battle_page(bid: str):
    p = f"battles/{bid}.html"
    return FileResponse(p) if os.path.exists(p) else HTMLResponse("<h1>Not Found</h1>", 404)

@app.post(WEBHOOK_PATH)
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return JSONResponse({"ok": True})

# ---------- Init ----------
telegram_app: Optional[Application] = None
webhook_task = None

@app.on_event("startup")
async def on_startup():
    global telegram_app, webhook_task
    log.info("Starting bot...")
    
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
