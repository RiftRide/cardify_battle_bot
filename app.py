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
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
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

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pfp-battle-bot")

VIDEO_ENABLED = False  # Disabled - Grok not using card images properly. Set to True to re-enable.
# VIDEO_ENABLED = bool(XAI_API_KEY)
if not VIDEO_ENABLED:
    log.info("Video generation disabled - focusing on HTML replays")

# ---------- FastAPI ----------
app = FastAPI()

os.makedirs("battles", exist_ok=True)
os.makedirs("cards", exist_ok=True)
os.makedirs("videos", exist_ok=True)

try:
    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    log.warning(f"Templates/static not found: {e}")

# ---------- SQLite storage ----------
DB_PATH = "battles.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Existing battles table
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
    
    # Store pending challenges - using user_id
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_challenges (
            user_id INTEGER PRIMARY KEY,
            opponent_username TEXT,
            timestamp TEXT
        )
        """
    )
    
    # Store uploaded cards - using user_id
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS uploaded_cards (
            user_id INTEGER PRIMARY KEY,
            card_data TEXT,
            timestamp TEXT
        )
        """
    )
    
    conn.commit()
    conn.close()


init_db()


# ---------- State persistence helpers ----------
def save_pending_challenge(user_id: int, opponent_username: str):
    """Save pending challenge to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO pending_challenges VALUES (?, ?, ?)",
        (user_id, opponent_username, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def load_pending_challenges() -> dict:
    """Load pending challenges from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id, opponent_username FROM pending_challenges")
    challenges = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    return challenges


def save_uploaded_card(user_id: int, card_data: dict):
    """Save uploaded card to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO uploaded_cards VALUES (?, ?, ?)",
        (user_id, json.dumps(card_data), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def load_uploaded_cards() -> dict:
    """Load uploaded cards from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id, card_data FROM uploaded_cards")
    cards = {}
    for row in c.fetchall():
        try:
            cards[row[0]] = json.loads(row[1])
        except json.JSONDecodeError:
            log.error(f"Failed to parse card data for user_id {row[0]}")
    conn.close()
    return cards


def clear_challenge(user_id: int):
    """Remove a challenge from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM pending_challenges WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def clear_card(user_id: int):
    """Remove a card from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM uploaded_cards WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


# ---------- In-memory state (loaded from DB on startup) ----------
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
    Create a side-by-side card display for Grok video generation.
    Places both card images next to each other so Grok can see both fighters clearly.
    """
    # Load both card images
    def load_card_image(card):
        try:
            img = Image.open(card["path"]).convert("RGB")
            return img
        except Exception as e:
            log.warning(f"Could not load card image: {e}")
            # Create a fallback colored rectangle
            img = Image.new("RGB", (500, 700), (50, 50, 50))
            return img
    
    card1_img = load_card_image(card1)
    card2_img = load_card_image(card2)
    
    # Get dimensions - assume cards are roughly the same size
    c1_width, c1_height = card1_img.size
    c2_width, c2_height = card2_img.size
    
    # Make cards the same height for consistency
    target_height = max(c1_height, c2_height)
    
    # Resize cards to same height while maintaining aspect ratio
    c1_aspect = c1_width / c1_height
    c2_aspect = c2_width / c2_height
    
    new_c1_width = int(target_height * c1_aspect)
    new_c2_width = int(target_height * c2_aspect)
    
    card1_img = card1_img.resize((new_c1_width, target_height), Image.Resampling.LANCZOS)
    card2_img = card2_img.resize((new_c2_width, target_height), Image.Resampling.LANCZOS)
    
    # Create canvas with small gap between cards
    gap = 40
    total_width = new_c1_width + gap + new_c2_width
    total_height = target_height
    
    # Create dark background
    scene = Image.new("RGB", (total_width, total_height), (15, 15, 25))
    
    # Paste both cards side by side
    scene.paste(card1_img, (0, 0))
    scene.paste(card2_img, (new_c1_width + gap, 0))
    
    # Convert to JPEG
    output = io.BytesIO()
    scene.save(output, format="JPEG", quality=95)
    output.seek(0)
    return output.getvalue()


# ---------- Grok Video Generation (FIXED WITH LONGER TIMEOUT) ----------
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
        
        # Determine winner and their final attack
        winner_name = winner_char if winner_char else name1  # Default to card1 if tie
        loser_name = name2 if winner_char == name1 else name1
        
        # Get winner's special ability for the final attack
        abilities1 = card1.get("abilities", [])
        abilities2 = card2.get("abilities", [])
        
        if winner_char == name1 and abilities1:
            final_attack = abilities1[0].get("name", "Ultimate Strike")
        elif winner_char == name2 and abilities2:
            final_attack = abilities2[0].get("name", "Ultimate Strike")
        else:
            final_attack = "Ultimate Strike"
        
        # Determine winner card's style/aesthetic for arena
        winner_desc = card1.get("description", "") if winner_char == name1 else card2.get("description", "")
        
        # Build the exact prompt you specified
        prompt = (
            f"Make card frames and stats fade or dissolve into the background to reveal a battle arena. "
            f"The battle arena must match the style of {winner_name}'s card ({winner_desc}). "
            f"First {name1} and {name2} charge their power moves before the battle begins. "
            f"Replicate the final attack from the winner: {winner_name} uses {final_attack} against {loser_name}. "
            f"The clash creates an explosion. "
            f"Out of the explosion are the words 'WINNER' and the name '{winner_name}'"
        )

        log.info(f"Video prompt: {prompt[:250]}...")

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
                    "duration": 6,
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

        # Step 2: Poll for completion (INCREASED TO 10 MINUTES with exponential backoff)
        max_wait = 600  # 10 minutes instead of 5
        elapsed = 0
        wait_time = 5  # Start with 5 seconds

        async with httpx.AsyncClient(timeout=30.0) as client:
            while elapsed < max_wait:
                await asyncio.sleep(wait_time)
                elapsed += wait_time

                try:
                    poll_response = await client.get(
                        f"https://api.x.ai/v1/videos/{request_id}",
                        headers={"Authorization": f"Bearer {XAI_API_KEY}"},
                    )

                    # 202 means still processing - this is OK, not an error
                    if poll_response.status_code == 202:
                        log.info(f"Video still processing... ({elapsed}s)")
                        wait_time = min(wait_time * 1.2, 30)
                        continue
                    
                    if poll_response.status_code != 200:
                        log.warning(f"Video poll unexpected status: {poll_response.status_code}")
                        wait_time = min(wait_time, 10)
                        continue

                    result = poll_response.json()
                    status = result.get("status", "pending")

                    log.info(f"Video status: {status} ({elapsed}s) - Full response: {result}")

                    if status == "done" or status == "completed":
                        # Try multiple possible response structures
                        video_url = None
                        
                        # Check different possible locations for the video URL
                        if "video" in result:
                            if isinstance(result["video"], dict):
                                video_url = result["video"].get("url") or result["video"].get("download_url")
                            elif isinstance(result["video"], str):
                                video_url = result["video"]
                        
                        if not video_url and "url" in result:
                            video_url = result["url"]
                        
                        if not video_url and "video_url" in result:
                            video_url = result["video_url"]
                        
                        if not video_url and "download_url" in result:
                            video_url = result["download_url"]
                        
                        if not video_url and "output" in result:
                            if isinstance(result["output"], dict):
                                video_url = result["output"].get("url") or result["output"].get("video_url")
                            elif isinstance(result["output"], str):
                                video_url = result["output"]
                        
                        if video_url:
                            log.info(f"Video ready! Downloading from {video_url[:80]}...")

                            # Download the video
                            video_response = await client.get(video_url, timeout=60.0)
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
                            log.error(f"Video done but no URL found. Full response keys: {list(result.keys())}")
                            log.error(f"Full response: {json.dumps(result, indent=2)}")
                            return None

                    elif status == "failed" or status == "expired":
                        log.error(f"Video generation failed with status: {status}")
                        return None

                    else:
                        log.info(f"Video still generating... ({elapsed}s)")
                        # Exponential backoff, capped at 30 seconds
                        wait_time = min(wait_time * 1.2, 30)
                
                except httpx.TimeoutException:
                    log.warning("Video check timed out, retrying...")
                    wait_time = 5  # Reset to short wait after timeout
                except Exception as e:
                    log.error(f"Error checking video status: {e}")
                    wait_time = 5

        log.error(f"Video generation timed out after {max_wait} seconds")
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
    elif ab_type == "defense":
        result["defense_boost"] = intensity * 0.4
        result["damage_reduction"] = min(0.5, intensity * 0.2)
    elif ab_type == "heal":
        result["heal_percent"] = min(0.5, intensity * 0.15)
    elif ab_type == "buff":
        result["stat_boost"] = intensity * 0.3
    elif ab_type == "debuff":
        result["stat_reduction"] = intensity * 0.25
    else:  # special
        result["damage_mult"] = 1.0 + (intensity * 0.4)
        result["special_effect"] = True

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

        if hp1 <= 0:
            break

    return max(0, hp1), max(0, hp2), hp1_start, hp2_start, battle_log


def execute_card_ability(ability: dict, attacker_atk: int, defender_def: float,
                         damage_scale: float, attacker_hp: int, defender_hp: int,
                         attacker_max: int, defender_max: int,
                         attacker_name: str, defender_name: str,
                         current_mult: float = 1.0) -> dict:
    """Execute a card's ability and return results"""
    ab_type = ability.get("type", "attack")
    ab_name = ability.get("name", "Unknown")
    emoji = ability.get("emoji", "✨")

    damage_dealt = 0
    next_mult = 1.0
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


# ---------- Battle HTML ----------
def save_battle_html(battle_id: str, ctx: dict):
    os.makedirs("battles", exist_ok=True)

    log_data = ctx.get("battle_log", [])
    c1s = ctx["card1_stats"]
    c2s = ctx["card2_stats"]
    n1 = ctx["card1_char_name"]
    n2 = ctx["card2_char_name"]
    
    # Copy card images to battles directory for the HTML to reference
    card1_img_path = ""
    card2_img_path = ""
    
    if ctx.get("card1_image") and os.path.exists(ctx["card1_image"]):
        import shutil
        card1_img_name = f"{battle_id}_card1.jpg"
        card1_img_dest = f"battles/{card1_img_name}"
        shutil.copy2(ctx["card1_image"], card1_img_dest)
        card1_img_path = card1_img_name
    
    if ctx.get("card2_image") and os.path.exists(ctx["card2_image"]):
        import shutil
        card2_img_name = f"{battle_id}_card2.jpg"
        card2_img_dest = f"battles/{card2_img_name}"
        shutil.copy2(ctx["card2_image"], card2_img_dest)
        card2_img_path = card2_img_name

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
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    background:#0a0a1e;
    color:#fff;
    font-family:'Segoe UI',Arial,sans-serif;
    padding:15px;
    overflow-x:hidden;
}}
/* Telegram Web App theme support */
body.tg-theme {{
    background: var(--tg-theme-bg-color, #0a0a1e);
    color: var(--tg-theme-text-color, #fff);
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
    overflow:visible;
}}
.fighter::before {{
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, 
        rgba(100,200,255,0.3), 
        rgba(255,100,200,0.3), 
        rgba(100,255,200,0.3), 
        rgba(255,200,100,0.3));
    background-size: 400% 400%;
    border-radius: 12px;
    z-index: -1;
    opacity: 0;
    animation: auraFlow 8s ease infinite;
    transition: opacity 0.3s;
}}
.fighter:hover::before {{
    opacity: 0.6;
}}
@keyframes auraFlow {{
    0%, 100% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
}}
.fighter.shake {{
    animation: shake 0.3s ease-in-out;
}}
.fighter.shake::before {{
    opacity: 1;
    animation: auraFlow 0.5s ease infinite;
}}
.fighter.hit {{
    animation: hitFlash 0.4s ease;
}}
.fighter.ko {{
    opacity:0.4;
    filter:grayscale(100%);
}}
.card-image {{
    width:100%;
    max-width:200px;
    height:auto;
    border-radius:8px;
    margin:0 auto 10px;
    display:block;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    transition: all 0.1s;
    animation: cardIdle 3s ease-in-out infinite;
}}
.fighter.shake .card-image {{
    filter: brightness(0.8);
    animation: none;
}}
.fighter.hit .card-image {{
    filter: brightness(1.5) saturate(1.5);
    animation: none;
}}
.fighter.ko .card-image {{
    filter: grayscale(100%) brightness(0.5);
    animation: none;
}}
/* Idle breathing animation */
@keyframes cardIdle {{
    0%, 100% {{ 
        transform: translateY(0) scale(1);
        filter: brightness(1);
    }}
    50% {{ 
        transform: translateY(-3px) scale(1.01);
        filter: brightness(1.05);
    }}
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
@keyframes abilityGlow {{
    0% {{ 
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        transform: scale(1);
    }}
    50% {{ 
        box-shadow: 0 0 30px 10px rgba(224,64,251,0.8), 0 0 60px 20px rgba(224,64,251,0.4);
        transform: scale(1.05);
    }}
    100% {{ 
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        transform: scale(1);
    }}
}}
.fighter.ability .card-image {{
    animation: abilityGlow 1s ease-in-out, cardPulse 0.3s ease-in-out 3;
}}
@keyframes cardPulse {{
    0%, 100% {{ transform: scale(1) rotate(0deg); }}
    25% {{ transform: scale(1.03) rotate(-2deg); }}
    75% {{ transform: scale(1.03) rotate(2deg); }}
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
    <div class="title">⚔️ {n1} vs {n2}</div>

    <div class="fighters">
        <div class="fighter" id="fighter1">
            {"<img src='/battles/" + card1_img_path + "' class='card-image' alt='" + n1 + "'>" if card1_img_path else ""}
            <div class="char-name">{n1}</div>
            <div class="owner">@{ctx['card1_name']}</div>
            <div class="desc">"{c1s.get('description','')}"</div>
            <div class="stats">
                <div class="stat"><span>⚔️ Atk</span><span>{c1s.get('effective_atk','?')}</span></div>
                <div class="stat"><span>🛡 Block</span><span>{c1s.get('def_rating','?')}%</span></div>
                <div class="stat"><span>✨</span><span class="rarity-{c1s['rarity'].lower().replace(' ','').replace('-','')}">{c1s['rarity']}</span></div>
                <div class="stat"><span>🎫</span><span>#{c1s['serial']}</span></div>
                <div class="stat"><span>✨ Ability</span><span>{c1s.get('ability_rate','')}%</span></div>
            </div>
            {ability_tags(c1s)}
        </div>
        <div class="vs-divider">VS</div>
        <div class="fighter" id="fighter2">
            {"<img src='/battles/" + card2_img_path + "' class='card-image' alt='" + n2 + "'>" if card2_img_path else ""}
            <div class="char-name">{n2}</div>
            <div class="owner">@{ctx['card2_name']}</div>
            <div class="desc">"{c2s.get('description','')}"</div>
            <div class="stats">
                <div class="stat"><span>⚔️ Atk</span><span>{c2s.get('effective_atk','?')}</span></div>
                <div class="stat"><span>🛡 Block</span><span>{c2s.get('def_rating','?')}%</span></div>
                <div class="stat"><span>✨</span><span class="rarity-{c2s['rarity'].lower().replace(' ','').replace('-','')}">{c2s['rarity']}</span></div>
                <div class="stat"><span>🎫</span><span>#{c2s['serial']}</span></div>
                <div class="stat"><span>✨ Ability</span><span>{c2s.get('ability_rate','')}%</span></div>
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
        <button id="btn-play" onclick="togglePlay()">▶ Play</button>
        <button onclick="skipToEnd()">Skip ⏭</button>
        <button id="sound-toggle" onclick="toggleSound()">🔊 Sound ON</button>
        <span class="speed-label">Speed:</span>
        <button id="speed1" class="active" onclick="setSpeed(1)">1x</button>
        <button id="speed2" onclick="setSpeed(2)">2x</button>
        <button id="speed3" onclick="setSpeed(3)">3x</button>
    </div>

    <div class="winner-banner" id="winner-banner">
        {'🏆 ' + winner_display + ' wins!' if winner_display != 'Tie' else '🤝 Tie!'}
    </div>

    <div class="battle-feed" id="battle-feed">
        <div class="feed-title">📜 Battle Log</div>
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

// Sound system
let audioContext = null;
let soundEnabled = true;

function initAudio() {{
    if (!audioContext) {{
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }}
}}

function playHitSound(isCritical = false) {{
    if (!soundEnabled || !audioContext) return;
    
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    if (isCritical) {{
        // Critical hit - explosive sound
        oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
        oscillator.frequency.exponentialRampToValueAtTime(100, audioContext.currentTime + 0.2);
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
    }} else {{
        // Normal hit - punch sound
        oscillator.frequency.setValueAtTime(200, audioContext.currentTime);
        oscillator.frequency.exponentialRampToValueAtTime(50, audioContext.currentTime + 0.1);
        gainNode.gain.setValueAtTime(0.2, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
    }}
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.3);
}}

function playAbilitySound() {{
    if (!soundEnabled || !audioContext) return;
    
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    // Magical ability sound
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(400, audioContext.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(800, audioContext.currentTime + 0.3);
    
    gainNode.gain.setValueAtTime(0.15, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
}}

function speakAbility(abilityName) {{
    if (!soundEnabled || !window.speechSynthesis) return;
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(abilityName);
    utterance.rate = 1.2;
    utterance.pitch = 1.1;
    utterance.volume = 0.7;
    
    window.speechSynthesis.speak(utterance);
}}

function toggleSound() {{
    soundEnabled = !soundEnabled;
    const btn = document.getElementById('sound-toggle');
    btn.textContent = soundEnabled ? '🔊 Sound ON' : '🔇 Sound OFF';
    if (soundEnabled) {{
        initAudio();
    }}
}}

// Auto-start the battle after a brief delay
setTimeout(() => {{
    initAudio(); // Initialize audio on first interaction
    play();
}}, 500);

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
    document.getElementById('btn-play').textContent = '⏸ Pause';
    stepForward();
}}

function pause() {{
    playing = false;
    document.getElementById('btn-play').textContent = '▶ Play';
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
    
    // Spawn energy particles around attacker
    spawnEnergyParticles(attackerFighter);

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
        spawnImpactParticles(targetFighter, entry.event);
        
        // Play hit sounds
        const isCrit = entry.event === 'critical' || entry.event === 'desperate';
        playHitSound(isCrit);
    }}
    
    // KO effect when HP reaches 0
    if (hp1 <= 0) {{
        document.getElementById('fighter1').classList.add('ko');
    }}
    if (hp2 <= 0) {{
        document.getElementById('fighter2').classList.add('ko');
    }}

    // Ability flash
    if (entry.event === 'ability' && entry.ability) {{
        showAbilityFlash(entry.ability.emoji + ' ' + entry.ability.name);
        
        // Play ability sound and speak the name
        playAbilitySound();
        speakAbility(entry.ability.name);
        
        // Add glow effect to the attacker's card
        const attacker = document.getElementById(attackerFighter);
        attacker.classList.remove('ability');
        void attacker.offsetWidth;
        attacker.classList.add('ability');
        setTimeout(() => attacker.classList.remove('ability'), 1000);
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

function spawnEnergyParticles(fighterId) {{
    const fighter = document.getElementById(fighterId);
    const rect = fighter.getBoundingClientRect();
    const numParticles = 3;
    
    for (let i = 0; i < numParticles; i++) {{
        const particle = document.createElement('div');
        particle.style.position = 'fixed';
        particle.style.width = '4px';
        particle.style.height = '4px';
        particle.style.borderRadius = '50%';
        particle.style.pointerEvents = 'none';
        particle.style.background = 'radial-gradient(circle, rgba(100,200,255,0.8), rgba(100,200,255,0))';
        particle.style.boxShadow = '0 0 6px rgba(100,200,255,0.6)';
        
        const startX = rect.left + rect.width/2 + (Math.random() - 0.5) * rect.width;
        const startY = rect.top + rect.height/2 + (Math.random() - 0.5) * rect.height;
        const angle = Math.random() * Math.PI * 2;
        const distance = 30 + Math.random() * 40;
        const endX = startX + Math.cos(angle) * distance;
        const endY = startY + Math.sin(angle) * distance - 30;
        
        particle.style.left = startX + 'px';
        particle.style.top = startY + 'px';
        particle.style.transition = 'all 1.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        particle.style.opacity = '1';
        
        document.body.appendChild(particle);
        
        setTimeout(() => {{
            particle.style.left = endX + 'px';
            particle.style.top = endY + 'px';
            particle.style.opacity = '0';
            particle.style.transform = 'scale(2)';
        }}, 10);
        
        setTimeout(() => particle.remove(), 1600);
    }}
}}

function spawnImpactParticles(targetId, event) {{
    const target = document.getElementById(targetId);
    const rect = target.getBoundingClientRect();
    const particleCount = event === 'critical' || event === 'desperate' ? 15 : 8;
    
    for (let i = 0; i < particleCount; i++) {{
        const particle = document.createElement('div');
        particle.style.position = 'fixed';
        particle.style.width = '6px';
        particle.style.height = '6px';
        particle.style.borderRadius = '50%';
        particle.style.pointerEvents = 'none';
        
        // Color based on event type
        if (event === 'critical') {{
            particle.style.background = 'radial-gradient(circle, #ffd740, #ff9500)';
        }} else if (event === 'desperate') {{
            particle.style.background = 'radial-gradient(circle, #ff5722, #ff1744)';
        }} else if (event === 'ability') {{
            particle.style.background = 'radial-gradient(circle, #e040fb, #aa00ff)';
        }} else {{
            particle.style.background = 'radial-gradient(circle, #ff6b6b, #ee5a6f)';
        }}
        
        const startX = rect.left + rect.width/2;
        const startY = rect.top + rect.height/2;
        const angle = (Math.PI * 2 * i) / particleCount;
        const velocity = 50 + Math.random() * 50;
        const endX = startX + Math.cos(angle) * velocity;
        const endY = startY + Math.sin(angle) * velocity;
        
        particle.style.left = startX + 'px';
        particle.style.top = startY + 'px';
        particle.style.transition = 'all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        
        document.body.appendChild(particle);
        
        // Animate
        setTimeout(() => {{
            particle.style.left = endX + 'px';
            particle.style.top = endY + 'px';
            particle.style.opacity = '0';
            particle.style.transform = 'scale(0)';
        }}, 10);
        
        setTimeout(() => particle.remove(), 700);
    }}
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
        const hp1Bar = document.getElementById('hp1-bar');
        const hp2Bar = document.getElementById('hp2-bar');
        hp1Bar.style.width = ((entry.hp1 / hp1Start) * 100) + '%';
        hp2Bar.style.width = ((entry.hp2 / hp2Start) * 100) + '%';
        if (entry.hp1 <= 0) hp1Bar.className = 'hp-bar dead';
        if (entry.hp2 <= 0) hp2Bar.className = 'hp-bar dead';
        document.getElementById('hp1-text').textContent = entry.hp1 + '/' + hp1Start;
        document.getElementById('hp2-text').textContent = entry.hp2 + '/' + hp2Start;
        document.getElementById('hp1-inner').textContent = entry.hp1;
        document.getElementById('hp2-inner').textContent = entry.hp2;

        const div = document.createElement('div');
        div.className = 'feed-entry visible ' + (entry.event || 'normal');
        const hpTag = ' <span class="hp-inline">[' + entry.hp1 + ' vs ' + entry.hp2 + ']</span>';
        div.innerHTML = entry.text + hpTag;
        document.getElementById('battle-feed').appendChild(div);

        currentStep++;
    }}
    document.getElementById('battle-feed').scrollTop = document.getElementById('battle-feed').scrollHeight;
    showWinner();
}}

// Telegram Web App initialization
if (window.Telegram && window.Telegram.WebApp) {{
    const tg = window.Telegram.WebApp;
    tg.ready();
    tg.expand(); // Expand to full height
    
    // Apply Telegram theme
    if (tg.themeParams) {{
        document.body.classList.add('tg-theme');
    }}
    
    // Enable closing confirmation
    tg.enableClosingConfirmation();
}}
</script>
</body></html>"""

    path = f"battles/{battle_id}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def persist_battle_record(battle_id, ch_user, ch_stats, opp_user, opp_stats, winner, html_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT OR REPLACE INTO battles VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            battle_id,
            datetime.now().isoformat(),
            ch_user,
            json.dumps(ch_stats),
            opp_user,
            json.dumps(opp_stats),
            winner or "",
            html_path,
        ),
    )
    conn.commit()
    conn.close()


def rarity_emoji(rarity: str) -> str:
    r = rarity.lower().replace(" ", "").replace("-", "")
    if r == "legendary":
        return "👑"
    elif r in ("ultrarare", "ultra-rare"):
        return "💎"
    elif r == "rare":
        return "⭐"
    else:
        return "🎴"


# ---------- Telegram handlers ----------
async def cmd_battle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to **PFP Battle**!\n\n"
        "🎮 **How to Battle:**\n"
        "1️⃣ `/challenge @username` - Challenge someone\n"
        "2️⃣ Upload your card image\n"
        "3️⃣ Opponent uploads their card\n"
        "4️⃣ Battle begins automatically!\n\n"
        "❓ **Other Commands:**\n"
        "🤓 `/analyze` - Preview card stats (no battle)\n"
        "📼 `/mystats` - Your win/loss record\n"
        "🥇 `/leaderboard` - Top 10 fighters\n\n"
        "💡 **Stats that matter:**\n"
        "⚔ Higher Power = more attack damage\n"
        "🛡 Higher Defense = better damage reduction\n"
        "⚡ Rarity gets stat multipliers\n"
        "💪 Lower serial numbers are stronger!\n"
        "✨ Abilities trigger based on your rarity\n"
        "💥 Crits + comebacks keep it unpredictable\n\n"
        "🔥 Good luck!",
        parse_mode="Markdown",
    )


async def cmd_challenge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Challenge a specific user to battle"""
    user_id = update.message.from_user.id  # Use user ID, not chat ID
    username = update.message.from_user.username or str(update.message.from_user.id)

    if not context.args:
        await update.message.reply_text(
            "Usage: `/challenge @username`\n"
            "Then upload your card image!",
            parse_mode="Markdown",
        )
        return

    opponent = context.args[0].replace("@", "").lower()
    
    # Save to both memory and database
    pending_challenges[user_id] = opponent
    save_pending_challenge(user_id, opponent)
    
    await update.message.reply_text(
        f"⚔️ Challenge issued to @{opponent}!\n"
        f"Now upload your card image to lock in your fighter."
    )


async def cmd_resetdb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear pending challenges and uploaded cards (admin only)"""
    # Put your Telegram username here (without @)
    ADMIN_USERNAMES = ["riftride", "urqkel"]  # Add your username(s) here
    
    username = update.message.from_user.username or ""
    
    if username.lower() not in [u.lower() for u in ADMIN_USERNAMES]:
        await update.message.reply_text("⛔ This command is admin-only.")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Clear pending challenges and uploaded cards tables
        c.execute("DELETE FROM pending_challenges")
        c.execute("DELETE FROM uploaded_cards")
        
        conn.commit()
        conn.close()
        
        # Clear in-memory state too
        global pending_challenges, uploaded_cards
        pending_challenges.clear()
        uploaded_cards.clear()
        
        await update.message.reply_text(
            "✅ **Database Reset Complete!**\n\n"
            "Cleared:\n"
            "• All pending challenges\n"
            "• All uploaded cards\n\n"
            "Battle history preserved. Ready for fresh battles!",
            parse_mode="Markdown"
        )
        log.info(f"Database reset by @{username}")
        
    except Exception as e:
        log.exception(f"Database reset error: {e}")
        await update.message.reply_text(f"❌ Error resetting database: {str(e)}")


async def cmd_mystats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show user's battle statistics"""
    username = (update.message.from_user.username or str(update.message.from_user.id)).lower()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM battles")
    battles = c.fetchall()
    conn.close()

    wins = 0
    losses = 0
    ties = 0
    total = 0

    for b in battles:
        ch_user = b[2].lower()
        opp_user = b[4].lower()
        winner = b[6].lower()

        if username in (ch_user, opp_user):
            total += 1
            if winner == username:
                wins += 1
            elif winner == "":
                ties += 1
            else:
                losses += 1

    if total == 0:
        await update.message.reply_text(
            f"📊 **Stats for @{username}**\n\n"
            f"No battles yet! Upload a card to start fighting!",
            parse_mode="Markdown",
        )
    else:
        winrate = (wins / total * 100) if total > 0 else 0
        await update.message.reply_text(
            f"📊 **Stats for @{username}**\n\n"
            f"🏆 Wins: {wins}\n"
            f"💀 Losses: {losses}\n"
            f"🤝 Ties: {ties}\n"
            f"📈 Win Rate: {winrate:.1f}%\n"
            f"⚔️ Total Battles: {total}",
            parse_mode="Markdown",
        )


async def cmd_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show top 10 players on the leaderboard"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM battles")
    battles = c.fetchall()
    conn.close()

    # Calculate stats for all players
    player_stats = {}
    
    for b in battles:
        ch_user = b[2].lower()
        opp_user = b[4].lower()
        winner = b[6].lower()
        
        # Initialize players
        for player in [ch_user, opp_user]:
            if player not in player_stats:
                player_stats[player] = {"wins": 0, "losses": 0, "ties": 0, "total": 0}
        
        # Update stats
        player_stats[ch_user]["total"] += 1
        player_stats[opp_user]["total"] += 1
        
        if winner == ch_user:
            player_stats[ch_user]["wins"] += 1
            player_stats[opp_user]["losses"] += 1
        elif winner == opp_user:
            player_stats[opp_user]["wins"] += 1
            player_stats[ch_user]["losses"] += 1
        elif winner == "":
            player_stats[ch_user]["ties"] += 1
            player_stats[opp_user]["ties"] += 1
    
    if not player_stats:
        await update.message.reply_text(
            "🏆 **LEADERBOARD**\n\n"
            "No battles yet! Be the first to fight!",
            parse_mode="Markdown"
        )
        return
    
    # Sort by wins, then by win rate
    leaderboard = []
    for player, stats in player_stats.items():
        winrate = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
        leaderboard.append({
            "username": player,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "ties": stats["ties"],
            "total": stats["total"],
            "winrate": winrate
        })
    
    # Sort by wins (primary), then winrate (secondary)
    leaderboard.sort(key=lambda x: (x["wins"], x["winrate"]), reverse=True)
    
    # Build leaderboard message
    msg = "🏆 **TOP 10 FIGHTERS**\n\n"
    
    medals = ["🥇", "🥈", "🥉"]
    
    for i, player in enumerate(leaderboard[:10]):
        rank = i + 1
        medal = medals[i] if i < 3 else f"{rank}."
        
        msg += (
            f"{medal} **@{player['username']}**\n"
            f"   Wins: {player['wins']} | W/L: {player['wins']}-{player['losses']}-{player['ties']} | "
            f"Rate: {player['winrate']:.1f}%\n\n"
        )
    
    msg += f"\n📊 Total Players: {len(player_stats)}\n"
    msg += f"⚔️ Total Battles: {len(battles)}"
    
    await update.message.reply_text(msg, parse_mode="Markdown")


# Global state for analyze mode
analyze_mode = {}  # user_id -> True
analyze_cooldown = {}  # user_id -> timestamp


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enter analyze mode - next image will be analyzed for battle stats"""
    user_id = update.message.from_user.id
    username = update.message.from_user.username or str(update.message.from_user.id)
    
    # Check cooldown (5 minutes between analyses)
    now = datetime.now()
    if user_id in analyze_cooldown:
        last_use = analyze_cooldown[user_id]
        cooldown_seconds = 300  # 5 minutes
        elapsed = (now - last_use).total_seconds()
        
        if elapsed < cooldown_seconds:
            remaining = int(cooldown_seconds - elapsed)
            minutes = remaining // 60
            seconds = remaining % 60
            await update.message.reply_text(
                f"⏳ **Analysis cooldown active!**\n\n"
                f"Please wait {minutes}m {seconds}s before analyzing another card.\n\n"
                f"💡 This prevents spam and saves AI costs.\n"
                f"Ready to battle? `/challenge @username`",
                parse_mode="Markdown"
            )
            return
    
    # Clear any pending challenge - analyze mode is separate from battle mode
    if user_id in pending_challenges:
        old_opponent = pending_challenges[user_id]
        pending_challenges.pop(user_id, None)
        clear_challenge(user_id)
        log.info(f"Cleared pending challenge for @{username} (was challenging @{old_opponent})")
    
    # Set analyze mode flag
    analyze_mode[user_id] = True
    
    await update.message.reply_text(
        "📊 **Card Analysis Mode**\n\n"
        "Upload a **BATTLE CARD** image to see:\n"
        "• Battle HP, Attack, Defense\n"
        "• Ability trigger rates\n"
        "• Special abilities\n\n"
        "⚠️ **Only upload battle card images!**\n"
        "Random images will be rejected.\n\n"
        "⚡ Upload your card now!",
        parse_mode="Markdown"
    )


async def debug_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log all unhandled messages for debugging"""
    if update.message:
        log.info(f"Unhandled message from {update.message.from_user.username}: {update.message.text}")


async def handler_card_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo/document uploads for card battles"""
    try:
        user_id = update.message.from_user.id  # Use user ID, not chat ID
        username = (update.message.from_user.username or str(update.message.from_user.id)).lower()

        # Check if this user has a pending challenge OR is in analyze mode
        # User has challenge if: they challenged someone OR someone challenged them
        user_challenged_someone = user_id in pending_challenges
        someone_challenged_user = any(
            challenged_username.lower() == username 
            for challenged_username in pending_challenges.values()
        )
        has_challenge = user_challenged_someone or someone_challenged_user
        in_analyze_mode = analyze_mode.get(user_id, False)
        
        # Ignore images if no context (user didn't challenge or use /analyze)
        if not has_challenge and not in_analyze_mode:
            log.info(f"Ignoring image from @{username} - no active challenge or analyze mode")
            return

        # Get photo
        if update.message.photo:
            file = await update.message.photo[-1].get_file()
        elif update.message.document:
            file = await update.message.document.get_file()
        else:
            return

        log.info(f"Photo from @{username} (user_id: {user_id})")

        msg = await update.message.reply_text("🔍 Analyzing card...")

        # Download and analyze
        file_bytes = await file.download_as_bytearray()
        stats = await analyze_card_with_claude(bytes(file_bytes))
        
        # Validate it's actually a battle card
        # If Claude returns very generic/default stats, it's probably not a card
        if (stats.get("name") in ["Unknown Warrior", "Unknown Fighter", "Fighter"] or
            (stats.get("power", 0) == 50 and stats.get("defense", 0) == 50)):
            
            await msg.edit_text(
                "❌ **Not a battle card!**\n\n"
                "This doesn't look like a PFP battle card.\n\n"
                "Please upload a proper battle card image with:\n"
                "• Character artwork\n"
                "• Stats (Power, Defense, etc.)\n"
                "• Abilities\n\n"
                "💡 Create cards using the card generator bot!",
                parse_mode="Markdown"
            )
            
            # Clear analyze mode if active
            analyze_mode.pop(user_id, None)
            return

        # Check if in analyze mode
        if analyze_mode.get(user_id, False):
            # Clear analyze mode
            analyze_mode.pop(user_id, None)
            
            # Set cooldown
            analyze_cooldown[user_id] = datetime.now()
            
            # Show detailed analysis with battle formula
            c_emj = rarity_emoji(stats["rarity"])
            
            # Calculate battle stats
            rarity_mult = get_rarity_multiplier(stats["rarity"])
            serial_mult = get_serial_multiplier(stats["serial"])
            effective_atk = calculate_attack(stats)
            def_rating = calculate_defense_rating(stats)
            hp = calculate_hp(stats)
            ability_chance = get_ability_trigger_chance(stats["rarity"])
            
            abilities_text = "\n".join([
                f"  • **{ab.get('name', '?')}** ({ab.get('type', '?')})\n"
                f"    _{ab.get('description', 'No description')}_"
                for ab in stats.get("abilities", [])[:3]
            ])
            
            analysis = (
                f"📊 **{stats['name']}**\n\n"
                f"{c_emj} **{stats['rarity']}** • Serial #{stats['serial']}\n"
                f"_{stats.get('description', 'No description')}_\n\n"
                f"**⚔️ Battle Stats:**\n"
                f"❤️ HP: {hp}\n"
                f"💥 Attack: {effective_atk}\n"
                f"🛡 Block: {int(def_rating * 100)}%\n"
                f"✨ Ability Rate: {int(ability_chance * 100)}%\n\n"
                f"**🎴 Special Abilities:**\n{abilities_text}\n\n"
                f"💡 Ready to fight? `/challenge @username`"
            )
            
            await msg.edit_text(analysis, parse_mode="Markdown")
            return

        # Save card image
        card_path = f"cards/{username}_{user_id}.jpg"
        with open(card_path, "wb") as f:
            f.write(file_bytes)

        card = {**stats, "username": username, "path": card_path}

        # Save to both memory and database FIRST (before checking challenges)
        uploaded_cards[user_id] = card
        save_uploaded_card(user_id, card)

        # Check if this user has a pending challenge
        opp_name = pending_challenges.get(user_id)

        # Check if we can trigger a battle
        triggered_pair = None
        
        log.info(f"Checking pairs for @{username} (user_id: {user_id})")
        log.info(f"  - This user challenged: {opp_name}")
        log.info(f"  - All pending challenges: {pending_challenges}")
        log.info(f"  - All uploaded cards: {[(uid, card['username']) for uid, card in uploaded_cards.items()]}")

        # FIRST: Check if someone challenged this user (they uploaded, we're waiting for them)
        for other_user_id, challenged_username in pending_challenges.items():
            if other_user_id == user_id:
                continue
            # If someone challenged ME and THEY have a card uploaded
            if username == challenged_username.lower() and other_user_id in uploaded_cards:
                triggered_pair = (other_user_id, user_id)
                log.info(f"✓ Match found: @{uploaded_cards[other_user_id]['username']} challenged @{username}")
                break

        # SECOND: If this user challenged someone, check if that person uploaded
        if not triggered_pair and opp_name:
            log.info(f"  - Looking for opponent '{opp_name}' in uploaded cards...")
            for other_user_id, other_card in uploaded_cards.items():
                if other_user_id == user_id:
                    continue
                other_username = other_card['username'].lower()
                target_username = opp_name.lower()
                log.info(f"    - Checking card from user_id {other_user_id}: username='{other_username}' vs target='{target_username}'")
                # If I challenged someone and THEY uploaded
                if other_username == target_username:
                    triggered_pair = (user_id, other_user_id)
                    log.info(f"✓ Match found: @{username} challenged @{opp_name}")
                    break

        # If no match found yet, show waiting message
        if not triggered_pair:
            log.info(f"✗ No pair found for @{username}")
            if opp_name:
                await msg.edit_text(f"✅ {card['name']} locked in!\nWaiting for @{opp_name}...")
            else:
                await msg.edit_text(f"✅ {card['name']} locked in!\n⏳ Waiting for opponent...")
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
            "card1_image": c1.get("path", ""), "card2_image": c2.get("path", ""),
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
        
        # This will work after domain is whitelisted
        kb = InlineKeyboardMarkup([
          [InlineKeyboardButton("🎬 Watch Replay", web_app=WebAppInfo(url=url))]
        ])
        
        
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
        
        # LIVE BATTLE COMMENTARY + FINAL RESULTS - Run in background to avoid blocking webhook
        asyncio.create_task(
            send_battle_commentary(update, c1, c2, log_data, hp1_start, hp2_start, 
                                 num_rounds, c1_emj, c2_emj, result, kb)
        )

        log.info(f"=== BATTLE DONE: winner={winner_username or 'Tie'} ===")

        # Generate AI video in background
        if VIDEO_ENABLED:
            asyncio.create_task(
                send_battle_video(update, c1, c2, log_data, winner_char, bid)
            )

        # Clean up state (both memory and database)
        clear_card(cid)
        clear_card(oid)
        clear_challenge(cid)
        clear_challenge(oid)  # Also clear opponent's challenge if they had one
        uploaded_cards.pop(cid, None)
        uploaded_cards.pop(oid, None)
        pending_challenges.pop(cid, None)
        pending_challenges.pop(oid, None)

    except Exception as e:
        log.exception(f"!!! ERROR for @{username}: {e}")
        try:
            await update.message.reply_text("❌ Error processing card. Try again.")
        except:
            pass


async def send_battle_commentary(update: Update, c1: dict, c2: dict, 
                                log_data: list, hp1_start: int, hp2_start: int,
                                num_rounds: int, c1_emj: str, c2_emj: str,
                                final_result: str, keyboard):
    """Background task: send live battle commentary with dramatic timing, then final results"""
    try:
        commentary = []
        
        # Opening
        commentary.append(
            f"⚔️ **BATTLE START!**\n"
            f"{c1_emj} {c1['name']} vs {c2_emj} {c2['name']}\n"
            f"HP: {hp1_start} vs {hp2_start}"
        )
        
        # Show key moments from the battle
        ability_moments = [e for e in log_data if e.get("event") == "ability"]
        crit_moments = [e for e in log_data if e.get("event") in ("critical", "desperate")]
        
        # First ability use
        if ability_moments:
            first_ability = ability_moments[0]
            ab_name = first_ability.get("ability", {}).get("name", "Special Move")
            emoji = first_ability.get("ability", {}).get("emoji", "✨")
            attacker = c1['name'] if first_ability['attacker'] == 1 else c2['name']
            commentary.append(f"{emoji} **{attacker}** unleashes *{ab_name}*!")
        
        # Big crit or comeback moment
        if crit_moments:
            best_moment = crit_moments[0]
            attacker = c1['name'] if best_moment['attacker'] == 1 else c2['name']
            if best_moment['event'] == 'desperate':
                commentary.append(f"🔥 **COMEBACK!** {attacker} refuses to fall!")
            else:
                commentary.append(f"💥 **CRITICAL HIT!** {attacker} lands a devastating blow!")
        
        # Mid-battle status (around round 5-10)
        mid_rounds = [e for e in log_data if 5 <= e.get('round', 0) <= 10]
        if mid_rounds:
            mid = mid_rounds[-1]
            hp1_pct = int((mid['hp1'] / hp1_start) * 100)
            hp2_pct = int((mid['hp2'] / hp2_start) * 100)
            commentary.append(f"⚡ Battle rages! HP: {hp1_pct}% vs {hp2_pct}%")
        
        # Final moments
        if num_rounds > 3:
            commentary.append(f"💫 After {num_rounds} intense rounds...")
        
        # Send commentary with delays for dramatic effect
        for i, comment in enumerate(commentary):
            if i > 0:
                await asyncio.sleep(1.5)  # Pause between messages
            await update.message.reply_text(comment, parse_mode="Markdown")
        
        # Final pause before results
        await asyncio.sleep(2)
        
        # FINAL RESULTS
        await update.message.reply_text(final_result, reply_markup=keyboard)
        
    except Exception as e:
        log.exception(f"Commentary error: {e}")


async def send_battle_video(update: Update, c1: dict, c2: dict,
                            log_data: list, winner_char: str, battle_id: str):
    """Background task: generate video and send when ready"""
    try:
        vid_msg = await update.message.reply_text(
            f"🎬 Generating 6-second AI battle video for {c1['name']} vs {c2['name']}...\n"
            f"⏳ Usually takes 5-10 minutes"
        )

        video_path = await generate_battle_video(c1, c2, log_data, winner_char, battle_id)

        if video_path and os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            log.info(f"Sending video: {video_path} ({file_size} bytes)")

            with open(video_path, "rb") as vf:
                await update.message.reply_video(
                    video=vf,
                    caption=f"⚔️ {c1['name']} vs {c2['name']} - AI Battle Clip",
                    supports_streaming=True,
                )

            await vid_msg.edit_text(
                f"✅ AI battle video ready!"
            )

            # Clean up
            try:
                os.remove(video_path)
            except Exception:
                pass
        else:
            await vid_msg.edit_text(
                f"🎬 Video didn't complete in time. "
                f"Check out the animated HTML replay above! ☝"
            )

    except TimeoutError:
        log.warning(f"Video generation timed out for battle {battle_id}")
        try:
            await vid_msg.edit_text(
                "🎬 Video took too long to generate. Use the HTML replay link!"
            )
        except Exception:
            pass
    except Exception as e:
        log.exception(f"Video send error: {e}")
        try:
            await vid_msg.edit_text("🎬 Video unavailable. Use the replay link above!")
        except Exception:
            pass

# ---------- FastAPI routes ----------
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ok", "bot": "PFP Battle", "vision": "Claude API"}


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"healthy": True, "status": "running"}


@app.api_route("/healthz", methods=["GET", "HEAD"]) 
async def healthz():
    """Alternative health check endpoint for Render"""
    return {"status": "ok"}


@app.get("/battle/{battle_id}")
async def battle_page(battle_id: str):
    path = f"battles/{battle_id}.html"
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return HTMLResponse("<h1>Battle Not Found</h1>", status_code=404)


@app.get("/battles/{filename}")
async def serve_battle_file(filename: str):
    """Serve card images and other files from battles directory"""
    path = f"battles/{filename}"
    if os.path.exists(path):
        # Determine media type
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            media_type = "image/jpeg"
        elif filename.endswith('.png'):
            media_type = "image/png"
        else:
            media_type = "application/octet-stream"
        return FileResponse(path, media_type=media_type)
    return HTMLResponse("<h1>File Not Found</h1>", status_code=404)


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
    global telegram_app, pending_challenges, uploaded_cards
    log.info("Starting bot...")
    
    # Load state from database
    pending_challenges = load_pending_challenges()
    uploaded_cards = load_uploaded_cards()
    log.info(f"Loaded {len(pending_challenges)} pending challenges")
    log.info(f"Loaded {len(uploaded_cards)} uploaded cards")

    telegram_app = Application.builder().token(BOT_TOKEN).build()
    telegram_app.add_handler(CommandHandler("battle", cmd_battle))
    telegram_app.add_handler(CommandHandler("start", cmd_battle))
    telegram_app.add_handler(CommandHandler("challenge", cmd_challenge))
    telegram_app.add_handler(CommandHandler("analyze", cmd_analyze))
    telegram_app.add_handler(CommandHandler("mystats", cmd_mystats))
    telegram_app.add_handler(CommandHandler("leaderboard", cmd_leaderboard))
    telegram_app.add_handler(CommandHandler("resetdb", cmd_resetdb))
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
            # Don't delete webhook on shutdown - this causes interruptions
            # The webhook will remain active for quick restarts
            await telegram_app.shutdown()
        except Exception as e:
            log.error(f"Shutdown error: {e}")
    log.info("Bot shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
