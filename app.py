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
import httpx
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Telegram imports
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from PIL import Image, ImageDraw, ImageFont
import anthropic

# ---------- Config ----------
BOT_TOKEN = os.getenv("BOT_TOKEN")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
PORT = int(os.getenv("PORT", 10000))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing in environment.")
if not RENDER_EXTERNAL_URL:
    raise RuntimeError("RENDER_EXTERNAL_URL missing in environment.")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY missing in environment.")

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = f"{RENDER_EXTERNAL_URL}{WEBHOOK_PATH}"

VIDEO_ENABLED = bool(XAI_API_KEY)
if not VIDEO_ENABLED:
    logging.getLogger("pfp-battle-bot").warning("XAI_API_KEY not set - video generation disabled")

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
os.makedirs("videos", exist_ok=True)

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
    "rare": 1.15,
    "ultrarare": 1.3,
    "ultra-rare": 1.3,
    "legendary": 1.5,
}

ABILITY_TRIGGER_CHANCE = {
    "common": 0.30,
    "rare": 0.22,
    "ultrarare": 0.15,
    "legendary": 0.10,
}


def get_rarity_multiplier(rarity: str) -> float:
    key = rarity.lower().replace(" ", "").replace("-", "")
    return RARITY_MULTIPLIER.get(key, 1.0)


def get_serial_multiplier(serial: int) -> float:
    if serial <= 0:
        serial = 1
    if serial <= 10:
        return 1.5
    elif serial <= 50:
        return 1.35
    elif serial <= 100:
        return 1.25
    elif serial <= 500:
        return 1.15
    elif serial <= 1000:
        return 1.08
    elif serial <= 5000:
        return 1.03
    else:
        return 1.0


def get_ability_trigger_chance(rarity: str) -> float:
    key = rarity.lower().replace(" ", "").replace("-", "")
    return ABILITY_TRIGGER_CHANCE.get(key, 0.25)


# ---------- Claude Vision ----------
claude_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


async def analyze_card_with_claude(file_bytes: bytes) -> dict:
    """Use Claude Vision to extract everything from the card including abilities"""
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

        log.info(f"Sending to Claude. Size: {len(file_bytes)} bytes, type: {media_type}")

        message = await asyncio.wait_for(
            claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
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
                                "text": """Analyze this PFP battle card and extract ALL information.

Look for:
1. Character Name - the name/title on the card
2. Power (attack stat) - EXACT number
3. Defense (defense stat) - EXACT number
4. Rarity (Common, Rare, Ultra-Rare, or Legendary)
5. Serial Number - labeled "Serial", "S/N", "#", or "422/1999" (use first number)
6. Description/Flavor text - any lore, backstory, or ability description
7. Special Abilities - ANY abilities, moves, or special powers mentioned on the card. Look for named attacks, spells, passive effects, or any special capability described in the card text.

Return ONLY valid JSON (no markdown, no code blocks):
{
  "name": "<character name>",
  "power": <number>,
  "defense": <number>,
  "rarity": "<rarity>",
  "serial": <number>,
  "description": "<flavor text or lore>",
  "abilities": [
    {"name": "<ability name>", "description": "<what it does>", "type": "<attack|defense|heal|buff|debuff|special>"}
  ]
}

IMPORTANT:
- Read EXACT values. Do NOT cap numbers.
- For abilities: extract EVERY ability, move, attack, or special power on the card.
- If the card has no explicit abilities, create 1-2 based on the character's appearance/theme/description.
- Each ability needs: name, short description of effect, and type.
- Type must be one of: attack, defense, heal, buff, debuff, special
- Defaults if not visible: name="Unknown Warrior", power=50, defense=50, rarity="Common", serial=1000"""
                            }
                        ],
                    }
                ],
            ),
            timeout=30.0
        )

        response_text = message.content[0].text.strip()
        log.info(f"Claude response: {response_text[:400]}")

        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()

        stats = json.loads(json_text)

        name = str(stats.get("name", "Unknown Warrior")).strip()[:50]
        if not name:
            name = "Unknown Warrior"

        power = max(1, int(stats.get("power", 50)))
        defense = max(1, int(stats.get("defense", 50)))
        rarity = str(stats.get("rarity", "Common"))
        serial = max(1, int(stats.get("serial", 1000)))
        description = str(stats.get("description", "A mysterious fighter.")).strip()[:200]
        if not description:
            description = "A mysterious fighter."

        raw_abilities = stats.get("abilities", [])
        abilities = []
        for ab in raw_abilities:
            if isinstance(ab, dict) and ab.get("name"):
                ab_type = str(ab.get("type", "attack")).lower()
                if ab_type not in ("attack", "defense", "heal", "buff", "debuff", "special"):
                    ab_type = "attack"
                abilities.append({
                    "name": str(ab["name"]).strip()[:40],
                    "description": str(ab.get("description", "")).strip()[:100],
                    "type": ab_type,
                })

        if not abilities:
            abilities = [{"name": "Strike", "description": "A basic attack", "type": "attack"}]

        log.info(f"Extracted: name={name}, power={power}, defense={defense}, "
                 f"rarity={rarity}, serial={serial}, abilities={len(abilities)}")

        return {
            "name": name, "power": power, "defense": defense,
            "rarity": rarity, "serial": serial,
            "description": description, "abilities": abilities,
        }

    except asyncio.TimeoutError:
        log.error("Claude API timed out")
        return _default_card_data()
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        return _default_card_data()
    except anthropic.APIError as e:
        log.error(f"Anthropic error: {e}")
        return _default_card_data()
    except Exception as e:
        log.exception(f"Claude error: {e}")
        return _default_card_data()


def _default_card_data():
    return {
        "name": "Unknown Warrior", "power": 50, "defense": 50,
        "rarity": "Common", "serial": 1000,
        "description": "A mysterious fighter.",
        "abilities": [{"name": "Strike", "description": "A basic attack", "type": "attack"}],
    }


# ---------- Battle scene image for video generation ----------
def create_battle_scene_image(card1: dict, card2: dict) -> bytes:
    """
    Combine both card images side by side into a battle scene.
    This image is sent to Grok Imagine Video as the starting frame.
    """
    WIDTH, HEIGHT = 1280, 720
    scene = Image.new("RGB", (WIDTH, HEIGHT), (10, 10, 30))
    draw = ImageDraw.Draw(scene)

    # Load and resize card images
    def load_card(card, fallback_color):
        try:
            img = Image.open(card["path"]).convert("RGB")
            img = img.resize((400, 400))
            return img
        except Exception:
            img = Image.new("RGB", (400, 400), fallback_color)
            return img

    card1_img = load_card(card1, (70, 130, 230))
    card2_img = load_card(card2, (230, 70, 70))

    # Place cards on left and right
    scene.paste(card1_img, (80, 160))
    scene.paste(card2_img, (800, 160))

    # Draw VS in center
    try:
        font_vs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
        font_name = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font_vs = ImageFont.load_default()
        font_name = ImageFont.load_default()

    draw.text((WIDTH // 2, HEIGHT // 2), "VS", fill=(255, 80, 80),
              font=font_vs, anchor="mm")

    # Draw names
    draw.text((280, 140), card1.get("name", "Fighter 1")[:20],
              fill=(255, 215, 100), font=font_name, anchor="mm")
    draw.text((1000, 140), card2.get("name", "Fighter 2")[:20],
              fill=(255, 215, 100), font=font_name, anchor="mm")

    # Add dramatic gradient overlay at edges
    for x in range(50):
        alpha = int(255 * (1 - x / 50))
        draw.line([(x, 0), (x, HEIGHT)], fill=(10, 10, 30))
        draw.line([(WIDTH - x, 0), (WIDTH - x, HEIGHT)], fill=(10, 10, 30))

    output = io.BytesIO()
    scene.save(output, format="JPEG", quality=90)
    output.seek(0)
    return output.getvalue()


# ---------- Grok Video Generation ----------
async def generate_battle_video(card1: dict, card2: dict, battle_log: list,
                                winner_char: str, battle_id: str) -> Optional[str]:
    """
    Generate an AI battle video using xAI's Grok Imagine Video API.
    Returns the local file path of the downloaded video, or None if failed.
    """
    if not XAI_API_KEY:
        log.info("Video generation skipped - no XAI_API_KEY")
        return None

    try:
        # Create the battle scene image
        scene_bytes = create_battle_scene_image(card1, card2)
        scene_b64 = base64.b64encode(scene_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{scene_b64}"

        # Build a dramatic battle prompt from the card data
        name1 = card1.get("name", "Fighter 1")
        name2 = card2.get("name", "Fighter 2")
        visual1 = card1.get("visual", "a warrior")
        visual2 = card2.get("visual", "a fighter")

        # Pick key battle moments for the prompt
        ability_moments = [e for e in battle_log if e.get("event") == "ability"]
        crit_moments = [e for e in battle_log if e.get("event") in ("critical", "desperate")]

        battle_description = f"{name1} ({visual1}) battles {name2} ({visual2}) in an epic arena."

        if ability_moments:
            ab = ability_moments[0].get("ability", {})
            battle_description += f" {name1 if ability_moments[0]['attacker'] == 1 else name2} unleashes {ab.get('name', 'a special attack')}."

        if crit_moments:
            battle_description += " Devastating critical strikes land."

        if winner_char:
            battle_description += f" {winner_char} emerges victorious with a final powerful blow."
        else:
            battle_description += " Both fighters collapse in a draw."

        prompt = (
            f"Anime battle animation: {battle_description} "
            f"Dynamic action scene with energy effects, impact flashes, "
            f"dramatic camera angles. Dark arena background with glowing effects. "
            f"Epic anime fight choreography."
        )

        log.info(f"Video prompt: {prompt[:200]}...")

        # Step 1: Start generation
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_response = await client.post(
                "https://api.x.ai/v1/videos/generations",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {XAI_API_KEY}",
                },
                json={
                    "model": "grok-imagine-video",
                    "prompt": prompt,
                    "image_url": image_url,
                    "duration": 5,
                    "aspect_ratio": "16:9",
                    "resolution": "480p",
                },
            )

            if start_response.status_code != 200:
                log.error(f"Video start failed: {start_response.status_code} {start_response.text}")
                return None

            request_id = start_response.json().get("request_id")
            if not request_id:
                log.error("No request_id in video response")
                return None

            log.info(f"Video generation started: {request_id}")

        # Step 2: Poll for completion (up to 5 minutes)
        max_wait = 300
        poll_interval = 5
        elapsed = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                poll_response = await client.get(
                    f"https://api.x.ai/v1/videos/{request_id}",
                    headers={"Authorization": f"Bearer {XAI_API_KEY}"},
                )

                if poll_response.status_code != 200:
                    log.warning(f"Video poll error: {poll_response.status_code}")
                    continue

                result = poll_response.json()
                status = result.get("status", "pending")

                if status == "done":
                    video_url = result.get("video", {}).get("url")
                    if video_url:
                        log.info(f"Video ready! Downloading from {video_url[:80]}...")

                        # Download the video
                        video_response = await client.get(video_url)
                        if video_response.status_code == 200:
                            video_path = f"videos/{battle_id}.mp4"
                            with open(video_path, "wb") as f:
                                f.write(video_response.content)
                            log.info(f"Video saved: {video_path} ({len(video_response.content)} bytes)")
                            return video_path
                        else:
                            log.error(f"Video download failed: {video_response.status_code}")
                            return None
                    else:
                        log.error("Video done but no URL")
                        return None

                elif status == "expired":
                    log.error("Video request expired")
                    return None

                else:
                    log.info(f"Video still generating... ({elapsed}s)")

        log.error("Video generation timed out after 5 minutes")
        return None

    except Exception as e:
        log.exception(f"Video generation error: {e}")
        return None

    
# ---------- Ability power calculation ----------
def calculate_ability_power(ability: dict, card: dict) -> dict:
    """Calculate ability effects based on type and keywords"""
    ab_type = ability.get("type", "attack")
    ab_desc = ability.get("description", "").lower()
    ab_name = ability.get("name", "").lower()
    
    power = card.get("power", 50)
    defense = card.get("defense", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))

    card_strength = (power + defense) / 2.0 * rarity_mult * serial_mult

    devastation_words = ["devastat", "ultimate", "supreme", "mega", "apocalyp", "annihilat",
                         "destroy", "obliterat", "godly", "divine", "omnislash", "realm"]
    strong_words = ["powerful", "mighty", "fierce", "brutal", "massive", "critical",
                    "crushing", "lethal", "deadly", "explosive", "raging", "furious"]
    moderate_words = ["solid", "steady", "reliable", "trained", "focused", "sharp",
                      "precise", "skilled", "enhanced", "charged"]
    weak_words = ["quick", "swift", "minor", "small", "light", "gentle", "weak",
                  "basic", "simple", "tiny"]

    combined_text = ab_name + " " + ab_desc
    
    if any(w in combined_text for w in devastation_words):
        intensity = 2.0
    elif any(w in combined_text for w in strong_words):
        intensity = 1.5
    elif any(w in combined_text for w in moderate_words):
        intensity = 1.2
    elif any(w in combined_text for w in weak_words):
        intensity = 0.7
    else:
        intensity = 1.0

    element_words = {
        "fire": "🔥", "flame": "🔥", "burn": "🔥", "inferno": "🔥",
        "ice": "❄️", "frost": "❄️", "freeze": "❄️", "cold": "❄️",
        "lightning": "⚡", "thunder": "⚡", "electric": "⚡", "shock": "⚡",
        "dark": "🌑", "shadow": "🌑", "void": "🌑", "death": "💀",
        "light": "✨", "holy": "✨", "divine": "✨", "sacred": "✨",
        "poison": "☠️", "toxic": "☠️", "venom": "☠️",
        "earth": "🌍", "stone": "🌍", "rock": "🌍",
        "wind": "💨", "air": "💨", "storm": "💨",
        "water": "🌊", "ocean": "🌊", "wave": "🌊",
        "psychic": "🔮", "mind": "🔮", "mental": "🔮",
    }

    emoji = "✨"
    for word, emj in element_words.items():
        if word in combined_text:
            emoji = emj
            break

    result = {
        "name": ability["name"],
        "description": ability.get("description", ""),
        "type": ab_type,
        "emoji": emoji,
        "intensity": intensity,
    }

    if ab_type == "attack":
        result["damage_mult"] = 1.0 + (intensity * 0.5)
        result["defense_bypass"] = min(0.6, intensity * 0.15)
        if any(w in combined_text for w in ["drain", "steal", "leech", "siphon", "absorb", "vampire"]):
            result["lifesteal_pct"] = 0.3
        if any(w in combined_text for w in ["flurry", "barrage", "multi", "rapid", "combo", "slash"]):
            result["hits"] = random.choice([2, 3])
        if any(w in combined_text for w in ["recoil", "sacrifice", "cost", "reckless", "kamikaze"]):
            result["self_damage_pct"] = 0.08

    elif ab_type == "defense":
        result["heal_pct"] = 0.03 + (intensity * 0.04)
        result["block_bonus"] = min(0.4, intensity * 0.12)

    elif ab_type == "heal":
        result["heal_pct"] = 0.05 + (intensity * 0.06)

    elif ab_type == "buff":
        result["next_attack_mult"] = 1.2 + (intensity * 0.3)

    elif ab_type == "debuff":
        result["enemy_miss_chance"] = min(0.4, 0.1 + intensity * 0.1)
        if any(w in combined_text for w in ["stun", "paralyze", "freeze", "petrif", "immobil"]):
            result["stun_rounds"] = 1

    elif ab_type == "special":
        if any(w in combined_text for w in ["stun", "paralyze", "freeze"]):
            result["stun_rounds"] = 1
            result["damage_mult"] = 1.0 + (intensity * 0.3)
        elif any(w in combined_text for w in ["heal", "restore", "regenerat"]):
            result["heal_pct"] = 0.08 + (intensity * 0.05)
        elif any(w in combined_text for w in ["shield", "protect", "barrier", "ward"]):
            result["heal_pct"] = 0.05
            result["block_bonus"] = 0.3
        else:
            result["damage_mult"] = 1.3 + (intensity * 0.4)
            result["defense_bypass"] = min(0.5, intensity * 0.2)

    return result


# ---------- Stat calculations ----------
def calculate_hp(card: dict) -> int:
    defense = card.get("defense", 50)
    power = card.get("power", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))
    base_hp = (defense * 0.7) + (power * 0.3)
    hp = int(base_hp * 3.0 * rarity_mult * serial_mult)
    return max(10, hp)


def calculate_attack(card: dict) -> int:
    power = card.get("power", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))
    attack = int(power * rarity_mult * serial_mult)
    return max(1, attack)


def calculate_defense_rating(card: dict) -> float:
    defense = card.get("defense", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))
    effective_def = defense * rarity_mult * serial_mult
    reduction = effective_def / (effective_def + 200.0)
    reduction = min(0.75, max(0.05, reduction))
    return round(reduction, 3)


# ---------- Battle simulation ----------
def simulate_battle(card1: dict, card2: dict):
    hp1 = calculate_hp(card1)
    hp2 = calculate_hp(card2)
    hp1_start = hp1
    hp2_start = hp2

    atk1 = calculate_attack(card1)
    atk2 = calculate_attack(card2)
    def1 = calculate_defense_rating(card1)
    def2 = calculate_defense_rating(card2)

    name1 = card1.get("name", "Fighter 1")
    name2 = card2.get("name", "Fighter 2")

    abilities1 = [calculate_ability_power(ab, card1) for ab in card1.get("abilities", [])]
    abilities2 = [calculate_ability_power(ab, card2) for ab in card2.get("abilities", [])]

    ability_chance1 = get_ability_trigger_chance(card1.get("rarity", "Common"))
    ability_chance2 = get_ability_trigger_chance(card2.get("rarity", "Common"))

    battle_log = []
    round_num = 0

    avg_hp = (hp1 + hp2) / 2.0
    avg_atk = (atk1 + atk2) / 2.0
    avg_def = (def1 + def2) / 2.0
    if avg_atk > 0:
        effective_avg = avg_atk * (1.0 - avg_def)
        if effective_avg > 0:
            damage_scale = avg_hp / (effective_avg * 14.0)
        else:
            damage_scale = 1.0
        damage_scale = max(0.01, min(damage_scale, 50.0))
    else:
        damage_scale = 1.0

    p1_next_mult = 1.0
    p2_next_mult = 1.0
    p1_stun = 0
    p2_stun = 0
    p1_miss_chance = 0.0
    p2_miss_chance = 0.0
    p1_temp_block = 0.0
    p2_temp_block = 0.0

    while hp1 > 0 and hp2 > 0 and round_num < 50:
        round_num += 1

        if round_num > 1:
            p1_miss_chance = max(0, p1_miss_chance - 0.15)
            p2_miss_chance = max(0, p2_miss_chance - 0.15)
            p1_temp_block = max(0, p1_temp_block - 0.1)
            p2_temp_block = max(0, p2_temp_block - 0.1)

        # Player 1 turn
        if p1_stun > 0:
            p1_stun -= 1
            battle_log.append({
                "round": round_num, "attacker": 1, "damage": 0,
                "event": "stunned", "ability": None,
                "hp1": max(0, hp1), "hp2": max(0, hp2),
                "text": f"💤 {name1} is stunned!"
            })
        else:
            if abilities1 and random.random() < ability_chance1:
                chosen = random.choice(abilities1)
                result = execute_card_ability(
                    chosen, atk1, def2 + p2_temp_block, damage_scale,
                    hp1, hp2, hp1_start, hp2_start, name1, name2, p1_next_mult
                )
                hp1 = result["attacker_hp"]
                hp2 = result["defender_hp"]
                p1_next_mult = result.get("next_mult", 1.0)
                p2_stun += result.get("stun_enemy", 0)
                p2_miss_chance = min(0.5, p2_miss_chance + result.get("enemy_miss", 0))
                p1_temp_block += result.get("self_block", 0)

                battle_log.append({
                    "round": round_num, "attacker": 1,
                    "damage": result.get("damage", 0),
                    "event": "ability", "ability": chosen,
                    "hp1": max(0, hp1), "hp2": max(0, hp2),
                    "text": result["text"]
                })
            else:
                if random.random() < p1_miss_chance:
                    battle_log.append({
                        "round": round_num, "attacker": 1, "damage": 0,
                        "event": "miss", "ability": None,
                        "hp1": max(0, hp1), "hp2": max(0, hp2),
                        "text": f"💨 {name1}'s attack misses!"
                    })
                else:
                    eff_def2 = min(0.75, def2 + p2_temp_block)
                    dmg, event = calculate_round_damage(
                        atk1, eff_def2, damage_scale,
                        hp1, hp2, hp1_start, hp2_start, p1_next_mult
                    )
                    p1_next_mult = 1.0
                    hp2 -= dmg
                    battle_log.append({
                        "round": round_num, "attacker": 1, "damage": dmg,
                        "event": event, "ability": None,
                        "hp1": max(0, hp1), "hp2": max(0, hp2),
                        "text": get_attack_text(name1, name2, dmg, event)
                    })

        if hp2 <= 0:
            break

        # Player 2 turn
        if p2_stun > 0:
            p2_stun -= 1
            battle_log.append({
                "round": round_num, "attacker": 2, "damage": 0,
                "event": "stunned", "ability": None,
                "hp1": max(0, hp1), "hp2": max(0, hp2),
                "text": f"💤 {name2} is stunned!"
            })
        else:
            if abilities2 and random.random() < ability_chance2:
                chosen = random.choice(abilities2)
                result = execute_card_ability(
                    chosen, atk2, def1 + p1_temp_block, damage_scale,
                    hp2, hp1, hp2_start, hp1_start, name2, name1, p2_next_mult
                )
                hp2 = result["attacker_hp"]
                hp1 = result["defender_hp"]
                p2_next_mult = result.get("next_mult", 1.0)
                p1_stun += result.get("stun_enemy", 0)
                p1_miss_chance = min(0.5, p1_miss_chance + result.get("enemy_miss", 0))
                p2_temp_block += result.get("self_block", 0)

                battle_log.append({
                    "round": round_num, "attacker": 2,
                    "damage": result.get("damage", 0),
                    "event": "ability", "ability": chosen,
                    "hp1": max(0, hp1), "hp2": max(0, hp2),
                    "text": result["text"]
                })
            else:
                if random.random() < p2_miss_chance:
                    battle_log.append({
                        "round": round_num, "attacker": 2, "damage": 0,
                        "event": "miss", "ability": None,
                        "hp1": max(0, hp1), "hp2": max(0, hp2),
                        "text": f"💨 {name2}'s attack misses!"
                    })
                else:
                    eff_def1 = min(0.75, def1 + p1_temp_block)
                    dmg, event = calculate_round_damage(
                        atk2, eff_def1, damage_scale,
                        hp2, hp1, hp2_start, hp1_start, p2_next_mult
                    )
                    p2_next_mult = 1.0
                    hp1 -= dmg
                    battle_log.append({
                        "round": round_num, "attacker": 2, "damage": dmg,
                        "event": event, "ability": None,
                        "hp1": max(0, hp1), "hp2": max(0, hp2),
                        "text": get_attack_text(name2, name1, dmg, event)
                    })

    return max(0, hp1), max(0, hp2), hp1_start, hp2_start, battle_log


def execute_card_ability(ability, attacker_atk, defender_def, damage_scale,
                         attacker_hp, defender_hp, attacker_max, defender_max,
                         attacker_name, defender_name, current_mult):
    """Execute an ability read from the actual card"""
    ab_type = ability.get("type", "attack")
    ab_name = ability.get("name", "Ability")
    emoji = ability.get("emoji", "✨")
    intensity = ability.get("intensity", 1.0)

    damage_dealt = 0
    next_mult = current_mult
    stun_enemy = 0
    enemy_miss = 0.0
    self_block = 0.0

    if ab_type == "attack":
        dmg_mult = ability.get("damage_mult", 1.5)
        def_bypass = ability.get("defense_bypass", 0.1)
        hits = ability.get("hits", 1)
        lifesteal = ability.get("lifesteal_pct", 0.0)
        self_dmg_pct = ability.get("self_damage_pct", 0.0)

        effective_def = max(0, defender_def * (1.0 - def_bypass))
        total_dmg = 0
        for _ in range(hits):
            base = attacker_atk * damage_scale * random.uniform(0.8, 1.2) * dmg_mult * current_mult
            hit_dmg = max(1, int(base * (1.0 - effective_def)))
            total_dmg += hit_dmg

        defender_hp -= total_dmg
        damage_dealt = total_dmg
        next_mult = 1.0

        if lifesteal > 0:
            heal = int(total_dmg * lifesteal)
            attacker_hp = min(attacker_max, attacker_hp + heal)
            text = f"{emoji} {attacker_name} uses {ab_name}! {total_dmg} dmg, steals {heal} HP!"
        elif self_dmg_pct > 0:
            self_hit = max(1, int(attacker_max * self_dmg_pct))
            attacker_hp -= self_hit
            text = f"{emoji} {attacker_name} uses {ab_name}! {total_dmg} dmg but takes {self_hit} recoil!"
        elif hits > 1:
            text = f"{emoji} {attacker_name} uses {ab_name}! {hits} hits for {total_dmg} total!"
        else:
            text = f"{emoji} {attacker_name} uses {ab_name}! {total_dmg} damage!"

    elif ab_type in ("defense",):
        heal_pct = ability.get("heal_pct", 0.06)
        block_bonus = ability.get("block_bonus", 0.15)
        heal = int(attacker_max * heal_pct)
        attacker_hp = min(attacker_max, attacker_hp + heal)
        self_block = block_bonus
        text = f"{emoji} {attacker_name} uses {ab_name}! Heals {heal} and boosts defense!"

    elif ab_type == "heal":
        heal_pct = ability.get("heal_pct", 0.12)
        heal = int(attacker_max * heal_pct)
        attacker_hp = min(attacker_max, attacker_hp + heal)
        text = f"{emoji} {attacker_name} uses {ab_name}! Restores {heal} HP!"

    elif ab_type == "buff":
        next_mult = ability.get("next_attack_mult", 1.5)
        text = f"{emoji} {attacker_name} uses {ab_name}! Next attack powered up {next_mult:.1f}x!"

    elif ab_type == "debuff":
        enemy_miss = ability.get("enemy_miss_chance", 0.2)
        stun_enemy = ability.get("stun_rounds", 0)
        if stun_enemy > 0:
            text = f"{emoji} {attacker_name} uses {ab_name}! {defender_name} is stunned!"
        else:
            text = f"{emoji} {attacker_name} uses {ab_name}! {defender_name} is weakened!"

    elif ab_type == "special":
        stun_enemy = ability.get("stun_rounds", 0)
        heal_pct = ability.get("heal_pct", 0)
        block_bonus = ability.get("block_bonus", 0)
        dmg_mult = ability.get("damage_mult", 0)
        def_bypass = ability.get("defense_bypass", 0)

        parts = []

        if dmg_mult > 0:
            effective_def = max(0, defender_def * (1.0 - def_bypass))
            base = attacker_atk * damage_scale * random.uniform(0.8, 1.2) * dmg_mult * current_mult
            hit_dmg = max(1, int(base * (1.0 - effective_def)))
            defender_hp -= hit_dmg
            damage_dealt = hit_dmg
            next_mult = 1.0
            parts.append(f"{hit_dmg} dmg")

        if heal_pct > 0:
            heal = int(attacker_max * heal_pct)
            attacker_hp = min(attacker_max, attacker_hp + heal)
            parts.append(f"heals {heal}")

        if block_bonus > 0:
            self_block = block_bonus
            parts.append(f"defense up")

        if stun_enemy > 0:
            parts.append(f"stuns {stun_enemy}rd")

        effect_text = ", ".join(parts) if parts else "mysterious effect"
        text = f"{emoji} {attacker_name} uses {ab_name}! {effect_text}!"
    else:
        text = f"{emoji} {attacker_name} uses {ab_name}!"

    return {
        "attacker_hp": attacker_hp, "defender_hp": defender_hp,
        "damage": damage_dealt, "next_mult": next_mult,
        "stun_enemy": stun_enemy, "enemy_miss": enemy_miss,
        "self_block": self_block, "text": text,
    }


def calculate_round_damage(attacker_atk, defender_def, damage_scale,
                           attacker_hp, defender_hp, attacker_max, defender_max,
                           attack_mult=1.0):
    base_dmg = attacker_atk * damage_scale * random.uniform(0.7, 1.3) * attack_mult

    attacker_hp_pct = attacker_hp / max(1, attacker_max)
    defender_hp_pct = defender_hp / max(1, defender_max)
    hp_disadvantage = max(0, defender_hp_pct - attacker_hp_pct)

    crit_chance = min(0.40, 0.15 + (hp_disadvantage * 0.5))
    roll = random.random()

    if roll < crit_chance:
        event = "desperate" if attacker_hp_pct < 0.3 else "critical"
        damage_after_def = base_dmg * 2.0 * (1.0 - defender_def * 0.5)
    elif roll < crit_chance + 0.10:
        event = "glancing"
        damage_after_def = base_dmg * 0.5 * (1.0 - defender_def)
    else:
        event = "normal"
        damage_after_def = base_dmg * (1.0 - defender_def)

    return max(1, int(damage_after_def)), event


def get_attack_text(attacker, defender, damage, event):
    if event == "desperate":
        return random.choice([
            f"🔥 {attacker} refuses to fall! {damage} damage!",
            f"🔥 Desperate strike from {attacker}! {damage}!",
        ])
    elif event == "critical":
        return random.choice([
            f"💥 {attacker} crits {defender} for {damage}!",
            f"💥 Critical hit! {damage} damage!",
        ])
    elif event == "glancing":
        return random.choice([
            f"💨 {attacker}'s attack glances off... {damage}.",
            f"💨 {defender} deflects! Only {damage}.",
        ])
    else:
        return random.choice([
            f"{attacker} hits {defender} for {damage}.",
            f"{attacker} strikes! {damage} to {defender}.",
        ])


# ---------- Battle HTML (UPDATED WITH TELEGRAM WEB APP) ----------
def save_battle_html(battle_id: str, ctx: dict):
    os.makedirs("battles", exist_ok=True)

    log_data = ctx.get("battle_log", [])
    c1s = ctx["card1_stats"]
    c2s = ctx["card2_stats"]
    n1 = ctx["card1_char_name"]
    n2 = ctx["card2_char_name"]

    if ctx["winner_name"] == ctx["card1_name"]:
        winner_display = n1
    elif ctx["winner_name"] == ctx["card2_name"]:
        winner_display = n2
    else:
        winner_display = "Tie"

    def ability_tags(stats):
        abilities = stats.get("abilities", [])
        if not abilities:
            return ""
        items = "".join(
            f'<span class="ab-tag">{a.get("name","?")}</span>' for a in abilities
        )
        return f'<div class="ab-tags">{items}</div>'

    # Build battle log as JSON for the JS animation
    log_json = json.dumps(log_data[:50])

    html = f"""<!DOCTYPE html>
<html><head><title>{n1} vs {n2}</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    background:#0a0a1e;
    color:#fff;
    font-family:'Segoe UI',Arial,sans-serif;
    padding:15px;
    overflow-x:hidden;
}}
.arena {{
    max-width:750px;
    margin:0 auto;
    position:relative;
}}
.title {{
    text-align:center;
    font-size:1.6em;
    margin-bottom:15px;
    background:linear-gradient(45deg,#ff6b6b,#ffd93d);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}}

/* Fighter cards */
.fighters {{
    display:flex;
    justify-content:space-between;
    gap:10px;
    margin-bottom:20px;
}}
.fighter {{
    flex:1;
    background:rgba(255,255,255,0.05);
    border-radius:12px;
    padding:12px;
    position:relative;
    transition: transform 0.1s;
}}
.fighter.shake {{
    animation: shake 0.3s ease-in-out;
}}
.fighter.hit {{
    animation: hitFlash 0.4s ease;
}}
.char-name {{
    font-size:1.2em;
    color:#ffd93d;
    font-weight:bold;
    text-align:center;
}}
.owner {{
    font-size:0.75em;
    color:#888;
    text-align:center;
    margin-bottom:6px;
}}
.desc {{
    font-size:0.7em;
    color:#aaa;
    font-style:italic;
    padding:4px 6px;
    background:rgba(0,0,0,0.2);
    border-radius:4px;
    margin-bottom:6px;
}}
.stats {{
    font-size:0.8em;
}}
.stat {{
    margin:2px 0;
    display:flex;
    justify-content:space-between;
}}
.ab-tags {{
    margin-top:6px;
    display:flex;
    flex-wrap:wrap;
    gap:3px;
}}
.ab-tag {{
    background:rgba(224,64,251,0.2);
    padding:1px 6px;
    border-radius:4px;
    font-size:0.7em;
    color:#e040fb;
}}
.vs-divider {{
    display:flex;
    align-items:center;
    font-size:2em;
    color:#ff6b6b;
    padding:0 5px;
}}

/* HP Bars */
.hp-section {{
    margin:15px 0;
}}
.hp-row {{
    margin:8px 0;
}}
.hp-label {{
    display:flex;
    justify-content:space-between;
    font-size:0.85em;
    margin-bottom:3px;
}}
.hp-bar-bg {{
    width:100%;
    height:28px;
    background:rgba(0,0,0,0.5);
    border-radius:14px;
    overflow:hidden;
    position:relative;
}}
.hp-bar {{
    height:100%;
    border-radius:14px;
    transition: width 0.6s ease-out;
    position:relative;
}}
.hp-bar.p1 {{ background:linear-gradient(90deg,#4CAF50,#8BC34A); }}
.hp-bar.p2 {{ background:linear-gradient(90deg,#2196F3,#03A9F4); }}
.hp-bar.dead {{ background:linear-gradient(90deg,#f44336,#ff5722); }}
.hp-text {{
    position:absolute;
    right:8px;
    top:50%;
    transform:translateY(-50%);
    font-size:0.8em;
    font-weight:bold;
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}}

/* Battle feed */
.battle-feed {{
    background:rgba(0,0,0,0.3);
    border-radius:12px;
    padding:15px;
    margin-top:15px;
    min-height:200px;
    max-height:350px;
    overflow-y:auto;
    scroll-behavior:smooth;
}}
.feed-title {{
    color:#ffd93d;
    margin-bottom:10px;
    font-size:1em;
}}
.feed-entry {{
    padding:6px 10px;
    margin:4px 0;
    border-radius:5px;
    font-size:0.85em;
    opacity:0;
    transform:translateX(-20px);
    transition: opacity 0.3s, transform 0.3s;
}}
.feed-entry.visible {{
    opacity:1;
    transform:translateX(0);
}}
.feed-entry.normal {{ border-left:3px solid #ff6b6b; background:rgba(255,107,107,0.05); }}
.feed-entry.critical {{ border-left:3px solid #ffd740; background:rgba(255,215,64,0.1); }}
.feed-entry.desperate {{ border-left:3px solid #ff5722; background:rgba(255,87,34,0.15); }}
.feed-entry.glancing {{ border-left:3px solid #666; background:rgba(100,100,100,0.08); }}
.feed-entry.ability {{ border-left:3px solid #e040fb; background:rgba(224,64,251,0.1); }}
.feed-entry.stunned {{ border-left:3px solid #00bcd4; background:rgba(0,188,212,0.08); }}
.feed-entry.miss {{ border-left:3px solid #999; background:rgba(150,150,150,0.05); }}
.hp-inline {{ color:#666; font-size:0.85em; }}

/* Floating damage */
.damage-float {{
    position:absolute;
    font-weight:bold;
    font-size:1.5em;
    pointer-events:none;
    animation:floatUp 1s ease-out forwards;
    z-index:10;
    text-shadow: 0 2px 4px rgba(0,0,0,0.8);
}}
.damage-float.crit {{ color:#ffd740; font-size:2em; }}
.damage-float.desperate {{ color:#ff5722; font-size:2.2em; }}
.damage-float.heal {{ color:#4CAF50; }}
.damage-float.ability {{ color:#e040fb; }}

/* Ability flash */
.ability-flash {{
    position:fixed;
    top:50%;
    left:50%;
    transform:translate(-50%,-50%);
    font-size:2em;
    font-weight:bold;
    color:#e040fb;
    text-shadow:0 0 20px rgba(224,64,251,0.8);
    pointer-events:none;
    animation:abilityPop 1s ease-out forwards;
    z-index:20;
    text-align:center;
    white-space:nowrap;
}}

/* Winner banner */
.winner-banner {{
    text-align:center;
    padding:15px;
    margin:15px 0;
    border-radius:12px;
    font-size:1.3em;
    font-weight:bold;
    opacity:0;
    transform:scale(0.5);
    transition: opacity 0.5s, transform 0.5s;
}}
.winner-banner.visible {{
    opacity:1;
    transform:scale(1);
}}
.winner-banner.p1win {{ background:linear-gradient(135deg,#4CAF50,#2E7D32); }}
.winner-banner.p2win {{ background:linear-gradient(135deg,#2196F3,#1565C0); }}
.winner-banner.tie {{ background:linear-gradient(135deg,#667eea,#764ba2); }}

/* Controls */
.controls {{
    display:flex;
    justify-content:center;
    gap:10px;
    margin:15px 0;
}}
.controls button {{
    background:rgba(255,255,255,0.1);
    border:1px solid rgba(255,255,255,0.2);
    color:#fff;
    padding:8px 20px;
    border-radius:8px;
    cursor:pointer;
    font-size:0.9em;
    transition:background 0.2s;
}}
.controls button:hover {{ background:rgba(255,255,255,0.2); }}
.controls button.active {{ background:rgba(224,64,251,0.3); border-color:#e040fb; }}
.speed-label {{ color:#888; font-size:0.8em; align-self:center; }}

/* Animations */
@keyframes shake {{
    0%,100% {{ transform:translateX(0); }}
    25% {{ transform:translateX(-8px); }}
    75% {{ transform:translateX(8px); }}
}}
@keyframes hitFlash {{
    0% {{ filter:brightness(1); }}
    50% {{ filter:brightness(2) saturate(2); }}
    100% {{ filter:brightness(1); }}
}}
@keyframes floatUp {{
    0% {{ opacity:1; transform:translateY(0); }}
    100% {{ opacity:0; transform:translateY(-60px); }}
}}
@keyframes abilityPop {{
    0% {{ opacity:0; transform:translate(-50%,-50%) scale(0.3); }}
    20% {{ opacity:1; transform:translate(-50%,-50%) scale(1.2); }}
    40% {{ transform:translate(-50%,-50%) scale(1.0); }}
    100% {{ opacity:0; transform:translate(-50%,-50%) scale(1.0) translateY(-30px); }}
}}
@keyframes screenShake {{
    0%,100% {{ transform:translate(0,0); }}
    10% {{ transform:translate(-5px,-3px); }}
    30% {{ transform:translate(5px,2px); }}
    50% {{ transform:translate(-3px,5px); }}
    70% {{ transform:translate(3px,-2px); }}
    90% {{ transform:translate(-2px,3px); }}
}}
.screen-shake {{
    animation:screenShake 0.4s ease-in-out;
}}

@media (max-width:600px) {{
    .fighters {{ flex-direction:column; }}
    .vs-divider {{ justify-content:center; font-size:1.5em; }}
    .char-name {{ font-size:1em; }}
    .ability-flash {{ font-size:1.3em; }}
}}
</style></head>
<body>
<div class="arena" id="arena">
    <div class="title">\u2694\ufe0f {n1} vs {n2}</div>

    <div class="fighters">
        <div class="fighter" id="fighter1">
            <div class="char-name">{n1}</div>
            <div class="owner">@{ctx['card1_name']}</div>
            <div class="desc">"{c1s.get('description','')}"</div>
            <div class="stats">
                <div class="stat"><span>\u2694\ufe0f Atk</span><span>{c1s.get('effective_atk','?')}</span></div>
                <div class="stat"><span>\U0001f6e1 Block</span><span>{c1s.get('def_rating','?')}%</span></div>
                <div class="stat"><span>\u2728</span><span class="rarity-{c1s['rarity'].lower().replace(' ','').replace('-','')}">{c1s['rarity']}</span></div>
                <div class="stat"><span>\U0001f3ab</span><span>#{c1s['serial']}</span></div>
                <div class="stat"><span>\u2728 Ability</span><span>{c1s.get('ability_rate','')}%</span></div>
            </div>
            {ability_tags(c1s)}
        </div>
        <div class="vs-divider">VS</div>
        <div class="fighter" id="fighter2">
            <div class="char-name">{n2}</div>
            <div class="owner">@{ctx['card2_name']}</div>
            <div class="desc">"{c2s.get('description','')}"</div>
            <div class="stats">
                <div class="stat"><span>\u2694\ufe0f Atk</span><span>{c2s.get('effective_atk','?')}</span></div>
                <div class="stat"><span>\U0001f6e1 Block</span><span>{c2s.get('def_rating','?')}%</span></div>
                <div class="stat"><span>\u2728</span><span class="rarity-{c2s['rarity'].lower().replace(' ','').replace('-','')}">{c2s['rarity']}</span></div>
                <div class="stat"><span>\U0001f3ab</span><span>#{c2s['serial']}</span></div>
                <div class="stat"><span>\u2728 Ability</span><span>{c2s.get('ability_rate','')}%</span></div>
            </div>
            {ability_tags(c2s)}
        </div>
    </div>

    <div class="hp-section">
        <div class="hp-row">
            <div class="hp-label">
                <span>{n1}</span>
                <span id="hp1-text">{ctx['hp1_start']}/{ctx['hp1_start']}</span>
            </div>
            <div class="hp-bar-bg">
                <div class="hp-bar p1" id="hp1-bar" style="width:100%">
                    <span class="hp-text" id="hp1-inner">{ctx['hp1_start']}</span>
                </div>
            </div>
        </div>
        <div class="hp-row">
            <div class="hp-label">
                <span>{n2}</span>
                <span id="hp2-text">{ctx['hp2_start']}/{ctx['hp2_start']}</span>
            </div>
            <div class="hp-bar-bg">
                <div class="hp-bar p2" id="hp2-bar" style="width:100%">
                    <span class="hp-text" id="hp2-inner">{ctx['hp2_start']}</span>
                </div>
            </div>
        </div>
    </div>

    <div class="controls">
        <button id="btn-play" onclick="togglePlay()">\u25b6 Play</button>
        <button onclick="skipToEnd()">Skip \u23ed</button>
        <span class="speed-label">Speed:</span>
        <button id="speed1" class="active" onclick="setSpeed(1)">1x</button>
        <button id="speed2" onclick="setSpeed(2)">2x</button>
        <button id="speed3" onclick="setSpeed(3)">3x</button>
    </div>

    <div class="winner-banner" id="winner-banner">
        {'\U0001f3c6 ' + winner_display + ' wins!' if winner_display != 'Tie' else '\U0001f91d Tie!'}
    </div>

    <div class="battle-feed" id="battle-feed">
        <div class="feed-title">\U0001f4dc Battle Log</div>
    </div>
</div>

<script>
const battleLog = {log_json};
const hp1Start = {ctx['hp1_start']};
const hp2Start = {ctx['hp2_start']};
const hp1End = {ctx['hp1_end']};
const hp2End = {ctx['hp2_end']};
const name1 = "{n1}";
const name2 = "{n2}";
const winnerName = "{winner_display}";

let currentStep = 0;
let playing = false;
let playTimer = null;
let speed = 1;
let baseDelay = 1200;

function setSpeed(s) {{
    speed = s;
    document.querySelectorAll('.controls button[id^=speed]').forEach(b => b.classList.remove('active'));
    document.getElementById('speed' + s).classList.add('active');
}}

function getDelay() {{
    return baseDelay / speed;
}}

function togglePlay() {{
    if (playing) {{
        pause();
    }} else {{
        play();
    }}
}}

function play() {{
    playing = true;
    document.getElementById('btn-play').textContent = '\u23f8 Pause';
    stepForward();
}}

function pause() {{
    playing = false;
    document.getElementById('btn-play').textContent = '\u25b6 Play';
    if (playTimer) clearTimeout(playTimer);
}}

function stepForward() {{
    if (currentStep >= battleLog.length) {{
        showWinner();
        pause();
        return;
    }}

    const entry = battleLog[currentStep];
    animateEntry(entry);
    currentStep++;

    if (playing) {{
        let delay = getDelay();
        if (entry.event === 'ability') delay *= 1.3;
        if (entry.event === 'desperate') delay *= 1.2;
        playTimer = setTimeout(stepForward, delay);
    }}
}}

function animateEntry(entry) {{
    // Update HP bars
    const hp1 = Math.max(0, entry.hp1);
    const hp2 = Math.max(0, entry.hp2);
    const hp1Pct = (hp1 / hp1Start) * 100;
    const hp2Pct = (hp2 / hp2Start) * 100;

    const hp1Bar = document.getElementById('hp1-bar');
    const hp2Bar = document.getElementById('hp2-bar');

    hp1Bar.style.width = hp1Pct + '%';
    hp2Bar.style.width = hp2Pct + '%';

    if (hp1 <= 0) hp1Bar.className = 'hp-bar dead';
    if (hp2 <= 0) hp2Bar.className = 'hp-bar dead';

    document.getElementById('hp1-text').textContent = hp1 + '/' + hp1Start;
    document.getElementById('hp2-text').textContent = hp2 + '/' + hp2Start;
    document.getElementById('hp1-inner').textContent = hp1;
    document.getElementById('hp2-inner').textContent = hp2;

    // Fighter hit animation
    const targetFighter = entry.attacker === 1 ? 'fighter2' : 'fighter1';
    const attackerFighter = entry.attacker === 1 ? 'fighter1' : 'fighter2';

    if (entry.damage > 0) {{
        const target = document.getElementById(targetFighter);
        target.classList.remove('shake', 'hit');
        void target.offsetWidth;
        target.classList.add('shake', 'hit');
        setTimeout(() => target.classList.remove('shake', 'hit'), 500);
    }}

    // Screen shake on crits
    if (entry.event === 'critical' || entry.event === 'desperate') {{
        const arena = document.getElementById('arena');
        arena.classList.remove('screen-shake');
        void arena.offsetWidth;
        arena.classList.add('screen-shake');
        setTimeout(() => arena.classList.remove('screen-shake'), 500);
    }}

    // Floating damage number
    if (entry.damage > 0) {{
        spawnDamageFloat(targetFighter, entry.damage, entry.event);
    }}

    // Ability flash
    if (entry.event === 'ability' && entry.ability) {{
        showAbilityFlash(entry.ability.emoji + ' ' + entry.ability.name);
    }}

    // Add to battle feed
    addFeedEntry(entry);
}}

function spawnDamageFloat(targetId, damage, event) {{
    const target = document.getElementById(targetId);
    const rect = target.getBoundingClientRect();
    const float = document.createElement('div');
    float.className = 'damage-float';
    if (event === 'critical') float.className += ' crit';
    if (event === 'desperate') float.className += ' desperate';
    if (event === 'ability') float.className += ' ability';

    float.textContent = '-' + damage;
    float.style.left = (rect.left + rect.width/2 - 20 + Math.random()*40) + 'px';
    float.style.top = (rect.top + 20 + Math.random()*30) + 'px';
    float.style.position = 'fixed';
    document.body.appendChild(float);
    setTimeout(() => float.remove(), 1100);
}}

function showAbilityFlash(text) {{
    const flash = document.createElement('div');
    flash.className = 'ability-flash';
    flash.textContent = text;
    document.body.appendChild(flash);
    setTimeout(() => flash.remove(), 1100);
}}

function addFeedEntry(entry) {{
    const feed = document.getElementById('battle-feed');
    const div = document.createElement('div');
    div.className = 'feed-entry ' + (entry.event || 'normal');

    const hpTag = ' <span class="hp-inline">[' + entry.hp1 + ' vs ' + entry.hp2 + ']</span>';
    div.innerHTML = entry.text + hpTag;

    feed.appendChild(div);

    requestAnimationFrame(() => {{
        div.classList.add('visible');
    }});

    feed.scrollTop = feed.scrollHeight;
}}

function showWinner() {{
    const banner = document.getElementById('winner-banner');
    if (winnerName === name1) banner.classList.add('p1win');
    else if (winnerName === name2) banner.classList.add('p2win');
    else banner.classList.add('tie');
    banner.classList.add('visible');
}}

function skipToEnd() {{
    pause();
    while (currentStep < battleLog.length) {{
        const entry = battleLog[currentStep];

        const hp1 = Math.max(0, entry.hp1);
        const hp2 = Math.max(0, entry.hp2);
        document.getElementById('hp1-bar').style.width = (hp1/hp1Start*100) + '%';
        document.getElementById('hp2-bar').style.width = (hp2/hp2Start*100) + '%';
        document.getElementById('hp1-text').textContent = hp1 + '/' + hp1Start;
        document.getElementById('hp2-text').textContent = hp2 + '/' + hp2Start;
        document.getElementById('hp1-inner').textContent = hp1;
        document.getElementById('hp2-inner').textContent = hp2;

        if (hp1 <= 0) document.getElementById('hp1-bar').className = 'hp-bar dead';
        if (hp2 <= 0) document.getElementById('hp2-bar').className = 'hp-bar dead';

        addFeedEntry(entry);
        currentStep++;
    }}
    showWinner();
}}

// Auto-play on load
setTimeout(() => play(), 800);
</script>
</body></html>"""

    path = f"battles/{battle_id}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def persist_battle_record(battle_id, challenger_username, challenger_stats,
                          opponent_username, opponent_stats, winner, html_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO battles (id, timestamp, challenger_username, challenger_stats, "
        "opponent_username, opponent_stats, winner, html_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (battle_id, datetime.utcnow().isoformat(), challenger_username,
         json.dumps(challenger_stats), opponent_username, json.dumps(opponent_stats),
         winner or "", html_path),
    )
    conn.commit()
    conn.close()


def rarity_emoji(rarity: str) -> str:
    key = rarity.lower().replace(" ", "").replace("-", "")
    return {"common": "⚪", "rare": "🔵",
            "ultrarare": "🟣", "legendary": "🟡"}.get(key, "✨")


# ---------- Telegram handlers ----------
async def cmd_battle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vid_status = "\U0001f3ac AI Video replays enabled!" if VIDEO_ENABLED else "\U0001f3ac Video replays disabled (no XAI key)"
    await update.message.reply_text(
        "⚔️ PFP Battle Bot\n\n"
        "/challenge @username - Start a battle\n"
        "/mystats - View your card stats\n\n"
        "💡 How battles work:\n"
        "⚔️ Power = attack damage\n"
        "🛡 Defense = % damage blocked\n"
        "🎴 Rarity & serial give a small edge\n"
        "🧐 Abilities are read from YOUR card!\n"
        "✨ Special abilities trigger randomly!\n"
        "💥 Crits + comebacks keep it unpredictable\n\n"
        f"{vid_status}\n"
        "🔒 HP hidden until both players upload!\n"
        "🤖 Powered by AI"
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
        "📤 Both players: upload your battle card image.\n"
        "🔒 Stats hidden until both cards are in!"
    )


async def cmd_mystats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    card = uploaded_cards.get(update.effective_user.id)
    if not card:
        await update.message.reply_text("❌ Upload a card first!")
        return

    hp = calculate_hp(card)
    atk = calculate_attack(card)
    def_r = calculate_defense_rating(card)
    r_emj = rarity_emoji(card["rarity"])
    ab_ch = get_ability_trigger_chance(card["rarity"])

    abilities = card.get("abilities", [])
    ab_text = ""
    if abilities:
        ab_lines = []
        for ab in abilities[:5]:
            computed = calculate_ability_power(ab, card)
            ab_lines.append(f"  {computed['emoji']} {ab['name']} ({ab['type']})")
        ab_text = "\n".join(ab_lines)

    await update.message.reply_text(
        f"📊 {card['name']}\n\n"
        f"⚔️ Power: {card['power']} → Attack: {atk}\n"
        f"🛡 Defense: {card['defense']} → Block: {int(def_r * 100)}%\n"
        f"{r_emj} {card['rarity']}\n"
        f"🎫 #{card['serial']}\n"
        f"❤️ HP: {hp}\n"
        f"✨ Ability rate: {int(ab_ch * 100)}%/round\n\n"
        f"💬 \"{card.get('description', '')}\"\n\n"
        f"Abilities:\n{ab_text}" if ab_text else ""
    )


async def debug_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    log.info(f"DEBUG: from @{update.effective_user.username} "
             f"photo={bool(update.message.photo)} doc={bool(update.message.document)}")


async def handler_card_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    username = (user.username or f"user{user.id}").lower()
    user_id = user.id

    log.info(f"=== UPLOAD START: @{username} (id:{user_id}) ===")

    try:
        file_obj = None
        if update.message.photo:
            log.info(f"Photo from @{username}")
            file_obj = await update.message.photo[-1].get_file()
        elif update.message.document:
            log.info(f"Document from @{username}: {update.message.document.mime_type}")
            file_obj = await update.message.document.get_file()
        else:
            return

        file_bytes = await file_obj.download_as_bytearray()
        if len(file_bytes) == 0:
            await update.message.reply_text("⚠️ Empty file.")
            return

        log.info(f"Downloaded {len(file_bytes)} bytes for @{username}")

        save_path = f"cards/{username}.png"
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        msg = await update.message.reply_text("🤖 Analyzing card...")
        log.info(f"Calling Claude for @{username}...")

        parsed = await analyze_card_with_claude(bytes(file_bytes))
        log.info(f"Claude done for @{username}: {parsed.get('name','?')} with {len(parsed.get('abilities',[]))} abilities")

        card = {
            "username": username,
            "user_id": user_id,
            "path": save_path,
            "name": parsed["name"],
            "power": int(parsed["power"]),
            "defense": int(parsed["defense"]),
            "rarity": parsed["rarity"],
            "serial": int(parsed["serial"]),
            "description": parsed.get("description", "A mysterious fighter."),
            "abilities": parsed.get("abilities", []),
        }

        uploaded_cards[user_id] = card
        log.info(f"Card stored for @{username}. Cards: {len(uploaded_cards)}, Challenges: {pending_challenges}")

        # Check challenge status
        in_challenge = False
        opponent_ready = False

        if user_id in pending_challenges:
            in_challenge = True
            opp_name = pending_challenges[user_id].lower()
            log.info(f"@{username} is challenger, looking for @{opp_name}")
            for uid, c in uploaded_cards.items():
                if c["username"].lower() == opp_name and uid != user_id:
                    opponent_ready = True
                    log.info(f"Opponent @{opp_name} found (id:{uid})")
                    break

        if not in_challenge:
            for cid, opp_name in pending_challenges.items():
                if username == opp_name.lower():
                    in_challenge = True
                    log.info(f"@{username} was challenged by id:{cid}")
                    if cid in uploaded_cards:
                        opponent_ready = True
                        log.info(f"Challenger id:{cid} has card")
                    break

        log.info(f"in_challenge={in_challenge}, opponent_ready={opponent_ready}")

        # HIDE if waiting
        if in_challenge and not opponent_ready:
            await msg.edit_text(
                f"✅ {card['name']} is locked in for @{username}!\n\n"
                f"🔒 Stats hidden until opponent uploads.\n"
                f"⏳ Waiting..."
            )
            return

        # NOT in challenge - show stats
        if not in_challenge:
            hp = calculate_hp(card)
            atk = calculate_attack(card)
            def_r = calculate_defense_rating(card)
            r_emj = rarity_emoji(card["rarity"])
            ab_ch = get_ability_trigger_chance(card["rarity"])

            ab_names = ", ".join(a["name"] for a in card.get("abilities", [])[:4])

            await msg.edit_text(
                f"✅ {card['name']} ready!\n"
                f"Owner: @{username}\n\n"
                f"⚔️ Atk: {card['power']} → {atk}\n"
                f"🛡 Block: {int(def_r * 100)}%\n"
                f"{r_emj} {card['rarity']} | 🎫 #{card['serial']}\n"
                f"❤️ HP: {hp}\n"
                f"✨ Ability rate: {int(ab_ch * 100)}%\n"
                f"💠 Moves: {ab_names}\n\n"
                f"💬 \"{card['description']}\"\n\n"
                f"Use /challenge @username to battle!"
            )
            return

        # BOTH READY - battle
        log.info("Both ready! Finding pair...")
        triggered_pair = None

        if user_id in pending_challenges:
            opp_name = pending_challenges[user_id].lower()
            opp_id = next(
                (uid for uid, c in uploaded_cards.items()
                 if c["username"].lower() == opp_name and uid != user_id),
                None,
            )
            if opp_id:
                triggered_pair = (user_id, opp_id)

        if not triggered_pair:
            for cid, opp_name in pending_challenges.items():
                if username == opp_name.lower() and cid in uploaded_cards:
                    triggered_pair = (cid, user_id)
                    break

        if not triggered_pair:
            log.warning(f"No pair found for @{username}")
            await msg.edit_text(f"✅ {card['name']} locked in!\n⏳ Waiting...")
            return

        cid, oid = triggered_pair
        c1, c2 = uploaded_cards[cid], uploaded_cards[oid]
        log.info(f"=== BATTLE: {c1['name']} vs {c2['name']} ===")

        hp1_end, hp2_end, hp1_start, hp2_start, log_data = simulate_battle(c1, c2)

        if hp1_end > hp2_end:
            winner_username = c1["username"]
            winner_char = c1["name"]
        elif hp2_end > hp1_end:
            winner_username = c2["username"]
            winner_char = c2["name"]
        else:
            winner_username = None
            winner_char = None

        bid = str(uuid.uuid4())

        def make_stats(card_data):
            return {
                "power": card_data["power"], "defense": card_data["defense"],
                "rarity": card_data["rarity"], "serial": card_data["serial"],
                "rarity_mult": get_rarity_multiplier(card_data["rarity"]),
                "serial_mult": get_serial_multiplier(card_data["serial"]),
                "effective_atk": calculate_attack(card_data),
                "def_rating": int(calculate_defense_rating(card_data) * 100),
                "description": card_data.get("description", ""),
                "abilities": card_data.get("abilities", []),
                "ability_rate": int(get_ability_trigger_chance(card_data["rarity"]) * 100),
            }

        battle_ctx = {
            "card1_name": c1["username"], "card2_name": c2["username"],
            "card1_char_name": c1["name"], "card2_char_name": c2["name"],
            "card1_stats": make_stats(c1), "card2_stats": make_stats(c2),
            "hp1_start": hp1_start, "hp2_start": hp2_start,
            "hp1_end": hp1_end, "hp2_end": hp2_end,
            "winner_name": winner_username or "Tie",
            "battle_id": bid, "battle_log": log_data,
        }

        html_path = save_battle_html(bid, battle_ctx)
        persist_battle_record(bid, c1["username"], battle_ctx["card1_stats"],
                              c2["username"], battle_ctx["card2_stats"],
                              winner_username, html_path)

        abilities_used = [e for e in log_data if e.get("event") == "ability"]
        crits = sum(1 for e in log_data if e.get("event") in ("critical", "desperate"))
        desperate = sum(1 for e in log_data if e.get("event") == "desperate")
        num_rounds = len(set(e["round"] for e in log_data))

        url = f"{RENDER_EXTERNAL_URL}/battle/{bid}"
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("\U0001f3ac Animated Replay", url=url)]])
        
        
        c1_emj = rarity_emoji(c1["rarity"])
        c2_emj = rarity_emoji(c2["rarity"])
        c1_ab_rate = int(get_ability_trigger_chance(c1["rarity"]) * 100)
        c2_ab_rate = int(get_ability_trigger_chance(c2["rarity"]) * 100)

        result = f"⚔️ {c1['name']} vs {c2['name']}\n\n"
        result += (
            f"{c1_emj} {c1['name']} (@{c1['username']})\n"
            f"Atk:{calculate_attack(c1)} Block:{int(calculate_defense_rating(c1)*100)}% "
            f"Ability:{c1_ab_rate}%\n"
            f"❤️ {hp1_end}/{hp1_start} HP\n\n"
        )
        result += (
            f"{c2_emj} {c2['name']} (@{c2['username']})\n"
            f"Atk:{calculate_attack(c2)} Block:{int(calculate_defense_rating(c2)*100)}% "
            f"Ability:{c2_ab_rate}%\n"
            f"❤️ {hp2_end}/{hp2_start} HP\n\n"
        )

        if winner_username:
            result += f"🏆 {winner_char} (@{winner_username}) wins!\n"
        else:
            result += "🤝 Tie!\n"

        result += f"⏱ {num_rounds} rounds"
        if crits > 0:
            result += f" | 💥 {crits} crits"
        if desperate > 0:
            result += f" | 🔥 {desperate} comebacks"
        if abilities_used:
            ab_names = list(set(
                e["ability"]["name"] for e in abilities_used if e.get("ability")
            ))
            if ab_names:
                result += f"\n✨ Moves used: {', '.join(ab_names[:6])}"

        await msg.edit_text(f"✅ {card['name']} locked in! Battle starting...")
        await update.message.reply_text(result, reply_markup=kb)
        log.info(f"=== BATTLE DONE: winner={winner_username or 'Tie'} ===")

        # Generate AI video in background
        if VIDEO_ENABLED:
            asyncio.create_task(
                send_battle_video(update, c1, c2, log_data, winner_char, bid)
            )

        uploaded_cards.pop(cid, None)
        uploaded_cards.pop(oid, None)
        pending_challenges.pop(cid, None)

    except Exception as e:
        log.exception(f"!!! ERROR for @{username}: {e}")
        try:
            await update.message.reply_text("❌ Error processing card. Try again.")
        except:
            pass


async def send_battle_video(update: Update, c1: dict, c2: dict,
                            log_data: list, winner_char: str, battle_id: str):
    """Background task: generate video and send when ready"""
    try:
        vid_msg = await update.message.reply_text(
            f"\U0001f3ac Generating AI battle video for {c1['name']} vs {c2['name']}...\n"
            f"\u23f3 This takes 1-3 minutes. Results are above \u261d"
        )

        video_path = await generate_battle_video(c1, c2, log_data, winner_char, battle_id)

        if video_path and os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            log.info(f"Sending video: {video_path} ({file_size} bytes)")

            with open(video_path, "rb") as vf:
                await update.message.reply_video(
                    video=vf,
                    caption=f"\u2694\ufe0f {c1['name']} vs {c2['name']} - AI Battle Replay",
                    supports_streaming=True,
                )

            await vid_msg.edit_text(
                f"\U0001f3ac AI battle video ready! \u261d"
            )

            # Clean up
            try:
                os.remove(video_path)
            except Exception:
                pass
        else:
            await vid_msg.edit_text(
                f"\U0001f3ac Video generation didn't complete. "
                f"Watch the animated replay instead! \u261d"
            )

    except Exception as e:
        log.exception(f"Video send error: {e}")
        try:
            await vid_msg.edit_text("\U0001f3ac Video unavailable. Use the replay link above!")
        except Exception:
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
    log.info("Webhook received")
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return JSONResponse({"ok": True})


# ---------- Startup / Shutdown ----------
telegram_app: Optional[Application] = None


@app.on_event("startup")
async def on_startup():
    global telegram_app
    log.info("Starting bot...")

    telegram_app = Application.builder().token(BOT_TOKEN).build()
    telegram_app.add_handler(CommandHandler("battle", cmd_battle))
    telegram_app.add_handler(CommandHandler("start", cmd_battle))
    telegram_app.add_handler(CommandHandler("challenge", cmd_challenge))
    telegram_app.add_handler(CommandHandler("mystats", cmd_mystats))
    telegram_app.add_handler(MessageHandler(filters.PHOTO, handler_card_upload))
    telegram_app.add_handler(MessageHandler(filters.Document.IMAGE, handler_card_upload))
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
