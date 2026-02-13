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

# ---------- Rarity & Serial config ----------
RARITY_MULTIPLIER = {
    "common": 1.0,
    "rare": 1.5,
    "ultrarare": 2.2,
    "ultra-rare": 2.2,
    "legendary": 3.0,
}

claude_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


async def analyze_card_with_claude(file_bytes: bytes) -> dict:
    """Use Claude Vision API to extract card stats - reads exact values"""
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
- Power (attack stat) - read the EXACT number shown, can be any value
- Defense (defense stat) - read the EXACT number shown, can be any value
- Rarity (usually Common, Rare, Ultra-Rare, or Legendary)
- Serial Number - usually labeled "Serial", "S/N", "#", or shown as "422/1999". Use the first number before any slash.

Return ONLY valid JSON (no markdown, no code blocks):
{"power": <exact number from card>, "defense": <exact number from card>, "rarity": "<exact rarity from card>", "serial": <exact serial number from card>}

IMPORTANT: Read the EXACT values printed on the card. Do NOT cap or limit numbers.
Only use defaults if a stat is truly not visible: power=50, defense=50, rarity="Common", serial=1000"""
                        }
                    ],
                }
            ],
        )

        response_text = message.content[0].text.strip()
        log.info(f"Claude response: {response_text[:200]}")

        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()

        stats = json.loads(json_text)

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


# ---------- Stat calculations ----------
def get_rarity_multiplier(rarity: str) -> float:
    key = rarity.lower().replace(" ", "").replace("-", "")
    return RARITY_MULTIPLIER.get(key, 1.0)


def get_serial_multiplier(serial: int) -> float:
    """Lower serial = rarer = stronger"""
    if serial <= 0:
        serial = 1
    if serial <= 10:
        return 2.0
    elif serial <= 50:
        return 1.7
    elif serial <= 100:
        return 1.5
    elif serial <= 500:
        return 1.3
    elif serial <= 1000:
        return 1.15
    elif serial <= 5000:
        return 1.05
    else:
        return 1.0


def calculate_hp(card: dict) -> int:
    """
    HP based on DEFENSE + rarity + serial.
    Tanky cards survive longer.
    
    HP = Defense * 3 * rarity_mult * serial_mult
    (Power is used for attack, not HP)
    """
    defense = card.get("defense", 50)
    power = card.get("power", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))

    # HP pool: defense-weighted with some power contribution
    # Defense = 70% weight, Power = 30% weight for HP
    base_hp = (defense * 0.7) + (power * 0.3)
    hp = int(base_hp * 3.0 * rarity_mult * serial_mult)
    return max(10, hp)


def calculate_attack(card: dict) -> int:
    """
    Attack based on POWER + rarity + serial.
    High power cards hit harder.
    """
    power = card.get("power", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))

    attack = int(power * rarity_mult * serial_mult)
    return max(1, attack)


def calculate_defense_rating(card: dict) -> float:
    """
    Defense rating = % of incoming damage blocked.
    Higher defense = more damage reduction.
    Scales with rarity and serial.
    
    Returns a value between 0.05 and 0.75 (5% to 75% damage blocked)
    """
    defense = card.get("defense", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))

    # Effective defense score
    effective_def = defense * rarity_mult * serial_mult

    # Convert to damage reduction percentage using diminishing returns
    # This prevents 100% damage block while still rewarding high defense
    # Formula: reduction = effective_def / (effective_def + 200)
    # At 50 def common: 50/(50+200) = 20% block
    # At 200 def legendary #1: (200*3*2)/(1200+200) = 85.7% -> capped at 75%
    reduction = effective_def / (effective_def + 200.0)
    reduction = min(0.75, max(0.05, reduction))

    return round(reduction, 3)


# ---------- Battle simulation ----------
def simulate_battle(card1: dict, card2: dict):
    """
    Turn-based battle with:
    - Power determines attack damage
    - Defense reduces incoming damage
    - Rarity & serial multiply everything
    - Critical hits (15% chance, 2x damage, bypasses 50% of defense)
    - Glancing blows (10% chance, 0.5x damage)
    - Underdog comeback: losing player gets increasing crit chance
    
    Returns (final_hp1, final_hp2, hp1_start, hp2_start, battle_log)
    """
    hp1 = calculate_hp(card1)
    hp2 = calculate_hp(card2)
    hp1_start = hp1
    hp2_start = hp2

    atk1 = calculate_attack(card1)
    atk2 = calculate_attack(card2)
    def1 = calculate_defense_rating(card1)
    def2 = calculate_defense_rating(card2)

    battle_log = []
    round_num = 0

    # Scale base damage so fights last 8-20 rounds
    avg_hp = (hp1 + hp2) / 2.0
    avg_atk = (atk1 + atk2) / 2.0
    avg_def_reduction = (def1 + def2) / 2.0

    if avg_atk > 0:
        effective_avg_dmg = avg_atk * (1.0 - avg_def_reduction)
        if effective_avg_dmg > 0:
            damage_scale = avg_hp / (effective_avg_dmg * 14.0)
        else:
            damage_scale = 1.0
        damage_scale = max(0.01, min(damage_scale, 50.0))
    else:
        damage_scale = 1.0

    while hp1 > 0 and hp2 > 0 and round_num < 50:
        round_num += 1

        # --- Player 1 attacks Player 2 ---
        dmg1, event1 = calculate_round_damage(
            atk1, def2, damage_scale, hp1, hp2, hp1_start, hp2_start
        )
        hp2 -= dmg1
        battle_log.append({
            "round": round_num,
            "attacker": 1,
            "damage": dmg1,
            "event": event1,
            "hp1": max(0, hp1),
            "hp2": max(0, hp2),
        })

        if hp2 <= 0:
            break

        # --- Player 2 attacks Player 1 ---
        dmg2, event2 = calculate_round_damage(
            atk2, def1, damage_scale, hp2, hp1, hp2_start, hp1_start
        )
        hp1 -= dmg2
        battle_log.append({
            "round": round_num,
            "attacker": 2,
            "damage": dmg2,
            "event": event2,
            "hp1": max(0, hp1),
            "hp2": max(0, hp2),
        })

    return max(0, hp1), max(0, hp2), hp1_start, hp2_start, battle_log


def calculate_round_damage(
    attacker_atk: int,
    defender_def: float,
    damage_scale: float,
    attacker_hp: int,
    defender_hp: int,
    attacker_max_hp: int,
    defender_max_hp: int,
) -> tuple[int, str]:
    """
    Calculate damage for one attack with randomness.
    
    Returns (damage_dealt, event_type)
    Event types: "normal", "critical", "glancing", "desperate"
    """
    # Base damage with variance
    base_dmg = attacker_atk * damage_scale * random.uniform(0.7, 1.3)

    # Determine event type
    roll = random.random()

    # Underdog comeback mechanic:
    # If attacker has less HP % than defender, crit chance increases
    attacker_hp_pct = attacker_hp / max(1, attacker_max_hp)
    defender_hp_pct = defender_hp / max(1, defender_max_hp)
    hp_disadvantage = max(0, defender_hp_pct - attacker_hp_pct)

    # Base crit chance 15%, up to 40% when losing badly
    crit_chance = 0.15 + (hp_disadvantage * 0.5)
    crit_chance = min(0.40, crit_chance)

    # Glancing blow chance 10%
    glancing_chance = 0.10

    event = "normal"

    if roll < crit_chance:
        # CRITICAL HIT - 2x damage, bypasses 50% of defense
        event = "critical"
        if attacker_hp_pct < 0.3:
            event = "desperate"  # Extra dramatic when near death
        effective_def = defender_def * 0.5  # Crits pierce half the armor
        damage_after_def = base_dmg * 2.0 * (1.0 - effective_def)
    elif roll < crit_chance + glancing_chance:
        # GLANCING BLOW - half damage
        event = "glancing"
        damage_after_def = base_dmg * 0.5 * (1.0 - defender_def)
    else:
        # NORMAL HIT
        damage_after_def = base_dmg * (1.0 - defender_def)

    # Minimum 1 damage always
    final_dmg = max(1, int(damage_after_def))

    return final_dmg, event


# ---------- Battle HTML ----------
def save_battle_html(battle_id: str, ctx: dict):
    """Generate battle replay HTML with event highlights."""
    os.makedirs("battles", exist_ok=True)

    log_html = ""
    for e in ctx.get("battle_log", [])[:30]:
        attacker = ctx["card1_name"] if e["attacker"] == 1 else ctx["card2_name"]
        event = e.get("event", "normal")

        if event == "critical":
            prefix = "\U0001f4a5 CRIT! "
            style = "border-left:3px solid #ffd740;background:rgba(255,215,64,0.1)"
        elif event == "desperate":
            prefix = "\U0001f525 DESPERATE CRIT! "
            style = "border-left:3px solid #ff5722;background:rgba(255,87,34,0.15)"
        elif event == "glancing":
            prefix = "\U0001f4a8 Glancing... "
            style = "border-left:3px solid #666;background:rgba(100,100,100,0.1)"
        else:
            prefix = ""
            style = "border-left:3px solid #ff6b6b;background:rgba(255,255,255,0.03)"

        log_html += (
            f'<div style="{style};padding:5px;margin:3px 0;border-radius:3px">'
            f'{prefix}R{e["round"]}: @{attacker} \u2192 {e["damage"]} dmg '
            f'(HP: {e["hp1"]} vs {e["hp2"]})</div>\n'
        )

    hp1_pct = max(0, int(ctx["hp1_end"] / max(1, ctx["hp1_start"]) * 100))
    hp2_pct = max(0, int(ctx["hp2_end"] / max(1, ctx["hp2_start"]) * 100))

    c1s = ctx["card1_stats"]
    c2s = ctx["card2_stats"]

    html = f"""<!DOCTYPE html>
<html><head><title>Battle {battle_id}</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body{{background:#0a0a1e;color:#fff;font-family:Arial;padding:20px;text-align:center}}
.arena{{background:rgba(255,255,255,0.05);border-radius:15px;padding:20px;margin:20px auto;max-width:750px}}
.fighters{{display:flex;justify-content:space-around;margin:20px 0;flex-wrap:wrap}}
.fighter{{flex:1;padding:10px;min-width:200px}}
.name{{font-size:1.3em;color:#ffd93d;margin-bottom:10px}}
.stats{{background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;text-align:left}}
.stat{{margin:4px 0;font-size:0.9em}}
.vs{{font-size:2.5em;color:#ff6b6b;margin:0 10px;align-self:center}}
.winner{{background:linear-gradient(135deg,#667eea,#764ba2);padding:15px;border-radius:10px;margin:15px 0;font-size:1.3em}}
.hp-section{{margin:15px 0}}
.hp-row{{margin:10px auto;text-align:left;max-width:500px}}
.hp-bar-bg{{width:100%;height:24px;background:rgba(0,0,0,0.5);border-radius:12px;overflow:hidden;margin-top:4px}}
.hp-bar{{height:100%;border-radius:12px;transition:width 2s ease-out}}
.hp-bar.green{{background:linear-gradient(90deg,#4CAF50,#8BC34A)}}
.hp-bar.red{{background:linear-gradient(90deg,#f44336,#ff5722)}}
.rarity-common{{color:#aaa}}.rarity-rare{{color:#4fc3f7}}.rarity-ultrarare{{color:#ba68c8}}.rarity-legendary{{color:#ffd740}}
.mult{{font-size:0.75em;color:#888}}
.log{{background:rgba(0,0,0,0.3);padding:15px;border-radius:10px;max-height:300px;overflow-y:auto;text-align:left;margin-top:20px}}
.legend{{display:flex;gap:15px;justify-content:center;margin:10px 0;font-size:0.8em;color:#888;flex-wrap:wrap}}
.legend span{{display:flex;align-items:center;gap:4px}}
</style></head><body>
<h1>\u2694\ufe0f Battle Replay</h1>
<div class="arena">
<div class="fighters">
<div class="fighter">
<div class="name">@{ctx['card1_name']}</div>
<div class="stats">
<div class="stat">\u2694\ufe0f Attack: {c1s['power']} <span class="mult">\u00d7{c1s.get('rarity_mult',1)} \u00d7{c1s.get('serial_mult',1)} = {c1s.get('effective_atk','?')}</span></div>
<div class="stat">\U0001f6e1 Defense: {c1s['defense']} <span class="mult">({c1s.get('def_rating','?')}% block)</span></div>
<div class="stat">\u2728 <span class="rarity-{c1s['rarity'].lower().replace(' ','').replace('-','')}">{c1s['rarity']}</span> <span class="mult">({c1s.get('rarity_mult','?')}x)</span></div>
<div class="stat">\U0001f3ab #{c1s['serial']} <span class="mult">({c1s.get('serial_mult','?')}x)</span></div>
<div class="stat">\u2764\ufe0f HP: {ctx['hp1_start']}</div>
</div></div>
<div class="vs">VS</div>
<div class="fighter">
<div class="name">@{ctx['card2_name']}</div>
<div class="stats">
<div class="stat">\u2694\ufe0f Attack: {c2s['power']} <span class="mult">\u00d7{c2s.get('rarity_mult',1)} \u00d7{c2s.get('serial_mult',1)} = {c2s.get('effective_atk','?')}</span></div>
<div class="stat">\U0001f6e1 Defense: {c2s['defense']} <span class="mult">({c2s.get('def_rating','?')}% block)</span></div>
<div class="stat">\u2728 <span class="rarity-{c2s['rarity'].lower().replace(' ','').replace('-','')}">{c2s['rarity']}</span> <span class="mult">({c2s.get('rarity_mult','?')}x)</span></div>
<div class="stat">\U0001f3ab #{c2s['serial']} <span class="mult">({c2s.get('serial_mult','?')}x)</span></div>
<div class="stat">\u2764\ufe0f HP: {ctx['hp2_start']}</div>
</div></div></div>
<div class="winner">{'\U0001f3c6 Winner: @' + ctx['winner_name'] if ctx['winner_name'] != 'Tie' else '\U0001f91d Tie!'}</div>
<div class="hp-section">
<div class="hp-row">
<span>@{ctx['card1_name']}: {ctx['hp1_end']}/{ctx['hp1_start']} HP</span>
<div class="hp-bar-bg"><div class="hp-bar {'green' if ctx['hp1_end'] > 0 else 'red'}" id="hp1" style="width:100%"></div></div>
</div>
<div class="hp-row">
<span>@{ctx['card2_name']}: {ctx['hp2_end']}/{ctx['hp2_start']} HP</span>
<div class="hp-bar-bg"><div class="hp-bar {'green' if ctx['hp2_end'] > 0 else 'red'}" id="hp2" style="width:100%"></div></div>
</div>
</div>
<div class="legend">
<span>\U0001f4a5 Critical Hit</span>
<span>\U0001f525 Desperate Crit</span>
<span>\U0001f4a8 Glancing Blow</span>
<span>\u2694\ufe0f Normal</span>
</div>
<div class="log"><h3>\U0001f4dc Battle Log</h3>{log_html}</div>
</div>
<script>
setTimeout(()=>{{
document.getElementById('hp1').style.width='{hp1_pct}%';
document.getElementById('hp2').style.width='{hp2_pct}%';
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


# ---------- Display helpers ----------
def rarity_emoji(rarity: str) -> str:
    key = rarity.lower().replace(" ", "").replace("-", "")
    return {
        "common": "\u26aa",
        "rare": "\U0001f535",
        "ultrarare": "\U0001f7e3",
        "legendary": "\U0001f7e1",
    }.get(key, "\u2728")


# ---------- Telegram handlers ----------
async def cmd_battle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\u2694\ufe0f PFP Battle Bot\n\n"
        "/challenge @username - Start a battle\n"
        "/mystats - View your card stats\n\n"
        "\U0001f4a1 How battles work:\n"
        "\u2022 \u2694\ufe0f Power = how hard you hit\n"
        "\u2022 \U0001f6e1 Defense = % damage blocked\n"
        "\u2022 \u2728 Rarity multiplies EVERYTHING\n"
        "\u2022 \U0001f3ab Low serial # = huge bonus\n"
        "\u2022 \U0001f4a5 Critical hits can turn the tide!\n"
        "\u2022 \U0001f525 Losing? Comeback crits kick in!\n\n"
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
        "\U0001f4e4 Both players: upload your battle card image.\n"
        "\U0001f512 Stats are hidden until both cards are in!"
    )


async def cmd_mystats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    card = uploaded_cards.get(update.effective_user.id)
    if not card:
        await update.message.reply_text("\u274c Upload a card first!")
        return

    hp = calculate_hp(card)
    atk = calculate_attack(card)
    def_rating = calculate_defense_rating(card)
    r_mult = get_rarity_multiplier(card["rarity"])
    s_mult = get_serial_multiplier(card["serial"])
    r_emj = rarity_emoji(card["rarity"])

    await update.message.reply_text(
        f"\U0001f4ca Your Card:\n\n"
        f"\u2694\ufe0f Power: {card['power']} \u2192 Attack: {atk}\n"
        f"\U0001f6e1 Defense: {card['defense']} \u2192 Blocks {int(def_rating * 100)}% dmg\n"
        f"{r_emj} Rarity: {card['rarity']} ({r_mult}x multiplier)\n"
        f"\U0001f3ab Serial: #{card['serial']} ({s_mult}x multiplier)\n"
        f"\u2764\ufe0f HP: {hp}\n\n"
        f"\U0001f4a1 Rarity & serial multiply attack, defense AND HP!"
    )


async def debug_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    log.info(
        f"DEBUG: Message from {update.effective_user.username} "
        f"photo={bool(update.message.photo)} doc={bool(update.message.document)} "
        f"text={bool(update.message.text)}"
    )


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

        # Check if in a pending challenge
        in_challenge = False
        opponent_ready = False

        if user_id in pending_challenges:
            in_challenge = True
            opp = pending_challenges[user_id].lower()
            opp_id = next(
                (uid for uid, c in uploaded_cards.items() if c["username"].lower() == opp),
                None,
            )
            if opp_id:
                opponent_ready = True
        else:
            for cid, opp in pending_challenges.items():
                if username == opp.lower():
                    in_challenge = True
                    if cid in uploaded_cards:
                        opponent_ready = True
                    break

        # HIDE STATS if in challenge and opponent hasn't uploaded
        if in_challenge and not opponent_ready:
            await msg.edit_text(
                f"\u2705 @{username}'s card is locked in!\n\n"
                f"\U0001f512 Stats hidden until opponent uploads.\n"
                f"\u23f3 Waiting for opponent..."
            )
            return

        # NOT in a challenge - safe to show full stats
        if not in_challenge:
            hp = calculate_hp(card)
            atk = calculate_attack(card)
            def_rating = calculate_defense_rating(card)
            r_mult = get_rarity_multiplier(card["rarity"])
            s_mult = get_serial_multiplier(card["serial"])
            r_emj = rarity_emoji(card["rarity"])

            await msg.edit_text(
                f"\u2705 @{username}'s card ready!\n\n"
                f"\u2694\ufe0f Power: {card['power']} \u2192 Attack: {atk}\n"
                f"\U0001f6e1 Defense: {card['defense']} \u2192 Blocks {int(def_rating * 100)}%\n"
                f"{r_emj} {card['rarity']} ({r_mult}x)\n"
                f"\U0001f3ab Serial #{card['serial']} ({s_mult}x)\n"
                f"\u2764\ufe0f HP: {hp}\n\n"
                f"\u2705 Use /challenge @username to battle!"
            )
            return

        # BOTH READY - run battle
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

        if not triggered_pair:
            await msg.edit_text(
                f"\u2705 @{username}'s card locked in!\n"
                f"\u23f3 Waiting for opponent..."
            )
            return

        cid, oid = triggered_pair
        c1, c2 = uploaded_cards[cid], uploaded_cards[oid]

        # Run battle
        hp1_end, hp2_end, hp1_start, hp2_start, log_data = simulate_battle(c1, c2)

        if hp1_end > hp2_end:
            winner = c1["username"]
        elif hp2_end > hp1_end:
            winner = c2["username"]
        else:
            winner = None

        # Calculate display stats
        c1_r_mult = get_rarity_multiplier(c1["rarity"])
        c1_s_mult = get_serial_multiplier(c1["serial"])
        c1_atk = calculate_attack(c1)
        c1_def = calculate_defense_rating(c1)

        c2_r_mult = get_rarity_multiplier(c2["rarity"])
        c2_s_mult = get_serial_multiplier(c2["serial"])
        c2_atk = calculate_attack(c2)
        c2_def = calculate_defense_rating(c2)

        bid = str(uuid.uuid4())
        battle_ctx = {
            "card1_name": c1["username"],
            "card2_name": c2["username"],
            "card1_stats": {
                "power": c1["power"],
                "defense": c1["defense"],
                "rarity": c1["rarity"],
                "serial": c1["serial"],
                "rarity_mult": c1_r_mult,
                "serial_mult": c1_s_mult,
                "effective_atk": c1_atk,
                "def_rating": int(c1_def * 100),
            },
            "card2_stats": {
                "power": c2["power"],
                "defense": c2["defense"],
                "rarity": c2["rarity"],
                "serial": c2["serial"],
                "rarity_mult": c2_r_mult,
                "serial_mult": c2_s_mult,
                "effective_atk": c2_atk,
                "def_rating": int(c2_def * 100),
            },
            "hp1_start": hp1_start,
            "hp2_start": hp2_start,
            "hp1_end": hp1_end,
            "hp2_end": hp2_end,
            "winner_name": winner or "Tie",
            "battle_id": bid,
            "battle_log": log_data,
        }

        html_path = save_battle_html(bid, battle_ctx)
        persist_battle_record(
            bid,
            c1["username"],
            battle_ctx["card1_stats"],
            c2["username"],
            battle_ctx["card2_stats"],
            winner,
            html_path,
        )

        # Count events
        crits = sum(1 for e in log_data if e.get("event") in ("critical", "desperate"))
        desperate = sum(1 for e in log_data if e.get("event") == "desperate")
        num_rounds = len(set(e["round"] for e in log_data))

        url = f"{RENDER_EXTERNAL_URL}/battle/{bid}"
        kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("\U0001f3ac View Full Replay", url=url)]]
        )

        # Build results message - REVEAL both cards
        c1_emj = rarity_emoji(c1["rarity"])
        c2_emj = rarity_emoji(c2["rarity"])

        result = f"\u2694\ufe0f Battle Complete!\n\n"

        result += (
            f"@{c1['username']}:\n"
            f"\u2694\ufe0f Atk:{c1_atk} \U0001f6e1 Block:{int(c1_def*100)}% "
            f"{c1_emj}{c1['rarity']} \U0001f3ab#{c1['serial']}\n"
            f"\u2764\ufe0f {hp1_end}/{hp1_start} HP\n\n"
        )

        result += (
            f"@{c2['username']}:\n"
            f"\u2694\ufe0f Atk:{c2_atk} \U0001f6e1 Block:{int(c2_def*100)}% "
            f"{c2_emj}{c2['rarity']} \U0001f3ab#{c2['serial']}\n"
            f"\u2764\ufe0f {hp2_end}/{hp2_start} HP\n\n"
        )

        if winner:
            result += f"\U0001f3c6 Winner: @{winner}!\n"
        else:
            result += "\U0001f91d It's a Tie!\n"

        result += f"\u23f1 {num_rounds} rounds"
        if crits > 0:
            result += f" | \U0001f4a5 {crits} crits"
        if desperate > 0:
            result += f" | \U0001f525 {desperate} comeback crits"

        await msg.edit_text(f"\u2705 @{username}'s card locked in! Battle starting...")
        await update.message.reply_text(result, reply_markup=kb)

        uploaded_cards.pop(cid, None)
        uploaded_cards.pop(oid, None)
        pending_challenges.pop(cid, None)

    except Exception as e:
        log.exception(f"Card upload error: {e}")
        try:
            await update.message.reply_text("\u274c Error processing card. Try again.")
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
