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
# Tighter multipliers - difference between tiers is smaller
# Common can compete, Legendary has an edge but not overwhelming
RARITY_MULTIPLIER = {
    "common": 1.0,
    "rare": 1.15,
    "ultrarare": 1.3,
    "ultra-rare": 1.3,
    "legendary": 1.5,
}

# Ability trigger chance - LOWER rarity = MORE triggers
# This is how common cards fight back: they spam abilities more often
ABILITY_TRIGGER_CHANCE = {
    "common": 0.30,      # 30% per round - very frequent
    "rare": 0.22,        # 22% per round
    "ultrarare": 0.15,   # 15% per round
    "legendary": 0.10,   # 10% per round - rare but powerful
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
  For example a fire character might have "Flame Burst" (attack) and "Heat Aura" (buff).
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

        # Parse abilities
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

        # Ensure at least one ability
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


# ---------- Ability power calculation ----------
def calculate_ability_power(ability: dict, card: dict) -> dict:
    """
    Intelligently determine what an ability is worth based on its type,
    description keywords, and the card's stats.
    
    Returns a dict with computed battle values for this ability.
    """
    ab_type = ability.get("type", "attack")
    ab_desc = ability.get("description", "").lower()
    ab_name = ability.get("name", "").lower()
    
    power = card.get("power", 50)
    defense = card.get("defense", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))

    # Base strength scales with card stats and multipliers
    card_strength = (power + defense) / 2.0 * rarity_mult * serial_mult

    # Scan description for power keywords
    # Devastating/ultimate/supreme/mega = very strong
    # Strong/powerful/mighty = strong
    # Quick/swift/minor = weaker but faster
    devastation_words = ["devastat", "ultimate", "supreme", "mega", "apocalyp", "annihilat",
                         "destroy", "obliterat", "godly", "divine", "omnislash", "realm"]
    strong_words = ["powerful", "mighty", "fierce", "brutal", "massive", "critical",
                    "crushing", "lethal", "deadly", "explosive", "raging", "furious"]
    moderate_words = ["solid", "steady", "reliable", "trained", "focused", "sharp",
                      "precise", "skilled", "enhanced", "charged"]
    weak_words = ["quick", "swift", "minor", "small", "light", "gentle", "weak",
                  "basic", "simple", "tiny"]

    # Determine intensity multiplier from description
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

    # Element/theme keywords for flavor (don't change power, just for logging)
    element_words = {
        "fire": "\U0001f525", "flame": "\U0001f525", "burn": "\U0001f525", "inferno": "\U0001f525",
        "ice": "\u2744\ufe0f", "frost": "\u2744\ufe0f", "freeze": "\u2744\ufe0f", "cold": "\u2744\ufe0f",
        "lightning": "\u26a1", "thunder": "\u26a1", "electric": "\u26a1", "shock": "\u26a1",
        "dark": "\U0001f311", "shadow": "\U0001f311", "void": "\U0001f311", "death": "\U0001f480",
        "light": "\u2728", "holy": "\u2728", "divine": "\u2728", "sacred": "\u2728",
        "poison": "\u2620\ufe0f", "toxic": "\u2620\ufe0f", "venom": "\u2620\ufe0f",
        "earth": "\U0001f30d", "stone": "\U0001f30d", "rock": "\U0001f30d",
        "wind": "\U0001f4a8", "air": "\U0001f4a8", "storm": "\U0001f4a8",
        "water": "\U0001f30a", "ocean": "\U0001f30a", "wave": "\U0001f30a",
        "psychic": "\U0001f52e", "mind": "\U0001f52e", "mental": "\U0001f52e",
    }

    emoji = "\u2728"
    for word, emj in element_words.items():
        if word in combined_text:
            emoji = emj
            break

    # Build the computed ability based on type
    result = {
        "name": ability["name"],
        "description": ability.get("description", ""),
        "type": ab_type,
        "emoji": emoji,
        "intensity": intensity,
    }

    if ab_type == "attack":
        # Damage = card power * intensity * small random factor
        # Higher intensity = more damage + some defense bypass
        result["damage_mult"] = 1.0 + (intensity * 0.5)  # 1.35 to 2.0
        result["defense_bypass"] = min(0.6, intensity * 0.15)  # 0.1 to 0.3
        # Check for lifesteal keywords
        if any(w in combined_text for w in ["drain", "steal", "leech", "siphon", "absorb", "vampire"]):
            result["lifesteal_pct"] = 0.3
        # Check for multi-hit
        if any(w in combined_text for w in ["flurry", "barrage", "multi", "rapid", "combo", "slash"]):
            result["hits"] = random.choice([2, 3])
        # Check for self-damage (recoil)
        if any(w in combined_text for w in ["recoil", "sacrifice", "cost", "reckless", "kamikaze"]):
            result["self_damage_pct"] = 0.08

    elif ab_type == "defense":
        result["heal_pct"] = 0.03 + (intensity * 0.04)  # 5.8% to 11%
        result["block_bonus"] = min(0.4, intensity * 0.12)  # temp extra block

    elif ab_type == "heal":
        result["heal_pct"] = 0.05 + (intensity * 0.06)  # 9.2% to 17%

    elif ab_type == "buff":
        result["next_attack_mult"] = 1.2 + (intensity * 0.3)  # 1.41 to 1.8

    elif ab_type == "debuff":
        result["enemy_miss_chance"] = min(0.4, 0.1 + intensity * 0.1)  # 17% to 30%
        # Check for stun
        if any(w in combined_text for w in ["stun", "paralyze", "freeze", "petrif", "immobil"]):
            result["stun_rounds"] = 1

    elif ab_type == "special":
        # Special abilities get a bit of everything based on keywords
        if any(w in combined_text for w in ["stun", "paralyze", "freeze"]):
            result["stun_rounds"] = 1
            result["damage_mult"] = 1.0 + (intensity * 0.3)
        elif any(w in combined_text for w in ["heal", "restore", "regenerat"]):
            result["heal_pct"] = 0.08 + (intensity * 0.05)
        elif any(w in combined_text for w in ["shield", "protect", "barrier", "ward"]):
            result["heal_pct"] = 0.05
            result["block_bonus"] = 0.3
        else:
            # Default special = strong attack with some bypass
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

    # Pre-compute ability power for each card's abilities
    abilities1 = [calculate_ability_power(ab, card1) for ab in card1.get("abilities", [])]
    abilities2 = [calculate_ability_power(ab, card2) for ab in card2.get("abilities", [])]

    ability_chance1 = get_ability_trigger_chance(card1.get("rarity", "Common"))
    ability_chance2 = get_ability_trigger_chance(card2.get("rarity", "Common"))

    battle_log = []
    round_num = 0

    # Damage scaling
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

    # Combat state
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

        # Decay debuffs
        if round_num > 1:
            p1_miss_chance = max(0, p1_miss_chance - 0.15)
            p2_miss_chance = max(0, p2_miss_chance - 0.15)
            p1_temp_block = max(0, p1_temp_block - 0.1)
            p2_temp_block = max(0, p2_temp_block - 0.1)

        # --- Player 1 turn ---
        if p1_stun > 0:
            p1_stun -= 1
            battle_log.append({
                "round": round_num, "attacker": 1, "damage": 0,
                "event": "stunned", "ability": None,
                "hp1": max(0, hp1), "hp2": max(0, hp2),
                "text": f"\U0001f4a4 {name1} is stunned!"
            })
        else:
            # Check ability trigger
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
                # Normal attack with miss check
                if random.random() < p1_miss_chance:
                    battle_log.append({
                        "round": round_num, "attacker": 1, "damage": 0,
                        "event": "miss", "ability": None,
                        "hp1": max(0, hp1), "hp2": max(0, hp2),
                        "text": f"\U0001f4a8 {name1}'s attack misses!"
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

        if hp2 <= 0:
            break

        # --- Player 2 turn ---
        if p2_stun > 0:
            p2_stun -= 1
            battle_log.append({
                "round": round_num, "attacker": 2, "damage": 0,
                "event": "stunned", "ability": None,
                "hp1": max(0, hp1), "hp2": max(0, hp2),
                "text": f"\U0001f4a4 {name2} is stunned!"
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
                        "text": f"\U0001f4a8 {name2}'s attack misses!"
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
    emoji = ability.get("emoji", "\u2728")
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
            f"\U0001f525 {attacker} refuses to fall! {damage} damage!",
            f"\U0001f525 Desperate strike from {attacker}! {damage}!",
        ])
    elif event == "critical":
        return random.choice([
            f"\U0001f4a5 {attacker} crits {defender} for {damage}!",
            f"\U0001f4a5 Critical hit! {damage} damage!",
        ])
    elif event == "glancing":
        return random.choice([
            f"\U0001f4a8 {attacker}'s attack glances off... {damage}.",
            f"\U0001f4a8 {defender} deflects! Only {damage}.",
        ])
    else:
        return random.choice([
            f"{attacker} hits {defender} for {damage}.",
            f"{attacker} strikes! {damage} to {defender}.",
        ])


# ---------- Battle HTML ----------
def save_battle_html(battle_id: str, ctx: dict):
    os.makedirs("battles", exist_ok=True)

    log_html = ""
    for e in ctx.get("battle_log", [])[:40]:
        event = e.get("event", "normal")
        text = e.get("text", "")

        styles = {
            "ability": "border-left:3px solid #e040fb;background:rgba(224,64,251,0.12)",
            "critical": "border-left:3px solid #ffd740;background:rgba(255,215,64,0.1)",
            "desperate": "border-left:3px solid #ff5722;background:rgba(255,87,34,0.15)",
            "glancing": "border-left:3px solid #666;background:rgba(100,100,100,0.1)",
            "stunned": "border-left:3px solid #00bcd4;background:rgba(0,188,212,0.1)",
            "miss": "border-left:3px solid #999;background:rgba(150,150,150,0.05)",
        }
        style = styles.get(event, "border-left:3px solid #ff6b6b;background:rgba(255,255,255,0.03)")

        hp_tag = f" <span style='color:#555;font-size:0.8em'>[{e['hp1']} vs {e['hp2']}]</span>"
        log_html += f'<div style="{style};padding:6px 8px;margin:3px 0;border-radius:3px">{text}{hp_tag}</div>\n'

    hp1_pct = max(0, int(ctx["hp1_end"] / max(1, ctx["hp1_start"]) * 100))
    hp2_pct = max(0, int(ctx["hp2_end"] / max(1, ctx["hp2_start"]) * 100))
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

    # Build ability list HTML
    def ability_html(stats):
        abilities = stats.get("abilities", [])
        if not abilities:
            return ""
        items = "".join(f'<span style="background:rgba(224,64,251,0.2);padding:2px 6px;border-radius:4px;margin:2px;display:inline-block;font-size:0.8em">{a.get("name","?")}</span>' for a in abilities)
        return f'<div style="margin-top:6px">{items}</div>'

    html = f"""<!DOCTYPE html>
<html><head><title>{n1} vs {n2}</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body{{background:#0a0a1e;color:#fff;font-family:'Segoe UI',Arial,sans-serif;padding:20px;text-align:center}}
.arena{{background:rgba(255,255,255,0.05);border-radius:15px;padding:20px;margin:20px auto;max-width:750px;box-shadow:0 4px 20px rgba(0,0,0,0.3)}}
.title{{font-size:1.8em;margin-bottom:5px;background:linear-gradient(45deg,#ff6b6b,#ffd93d);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.fighters{{display:flex;justify-content:space-around;margin:20px 0;flex-wrap:wrap;gap:10px}}
.fighter{{flex:1;padding:10px;min-width:220px}}
.char-name{{font-size:1.4em;color:#ffd93d;margin-bottom:2px;font-weight:bold}}
.owner{{font-size:0.85em;color:#888;margin-bottom:8px}}
.desc{{font-size:0.8em;color:#aaa;font-style:italic;margin:6px 0;padding:6px;background:rgba(0,0,0,0.2);border-radius:5px;text-align:left}}
.stats{{background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;text-align:left}}
.stat{{margin:4px 0;font-size:0.9em}}
.vs{{font-size:2.5em;color:#ff6b6b;margin:0 10px;align-self:center}}
.winner{{background:linear-gradient(135deg,#667eea,#764ba2);padding:15px;border-radius:10px;margin:15px 0;font-size:1.3em;box-shadow:0 4px 15px rgba(102,126,234,0.3)}}
.hp-section{{margin:15px 0}}
.hp-row{{margin:10px auto;text-align:left;max-width:500px}}
.hp-bar-bg{{width:100%;height:24px;background:rgba(0,0,0,0.5);border-radius:12px;overflow:hidden;margin-top:4px}}
.hp-bar{{height:100%;border-radius:12px;transition:width 2s ease-out}}
.hp-bar.green{{background:linear-gradient(90deg,#4CAF50,#8BC34A)}}
.hp-bar.red{{background:linear-gradient(90deg,#f44336,#ff5722)}}
.rarity-common{{color:#aaa}}.rarity-rare{{color:#4fc3f7}}.rarity-ultrarare{{color:#ba68c8}}.rarity-legendary{{color:#ffd740}}
.mult{{font-size:0.75em;color:#888}}
.log{{background:rgba(0,0,0,0.3);padding:15px;border-radius:10px;max-height:400px;overflow-y:auto;text-align:left;margin-top:20px}}
.legend{{display:flex;gap:12px;justify-content:center;margin:10px 0;font-size:0.75em;color:#888;flex-wrap:wrap}}
</style></head><body>
<div class="arena">
<div class="title">\u2694\ufe0f {n1} vs {n2}</div>
<div class="fighters">
<div class="fighter">
<div class="char-name">{n1}</div>
<div class="owner">@{ctx['card1_name']}</div>
<div class="desc">"{c1s.get('description','')}"</div>
<div class="stats">
<div class="stat">\u2694\ufe0f Atk: {c1s['power']} <span class="mult">\u2192 {c1s.get('effective_atk','?')}</span></div>
<div class="stat">\U0001f6e1 Def: {c1s['defense']} <span class="mult">({c1s.get('def_rating','?')}% block)</span></div>
<div class="stat">\u2728 <span class="rarity-{c1s['rarity'].lower().replace(' ','').replace('-','')}">{c1s['rarity']}</span></div>
<div class="stat">\U0001f3ab #{c1s['serial']}</div>
<div class="stat">\u2764\ufe0f {ctx['hp1_start']} HP</div>
<div class="stat">\u2728 Ability rate: {c1s.get('ability_rate','')}%</div>
{ability_html(c1s)}
</div></div>
<div class="vs">VS</div>
<div class="fighter">
<div class="char-name">{n2}</div>
<div class="owner">@{ctx['card2_name']}</div>
<div class="desc">"{c2s.get('description','')}"</div>
<div class="stats">
<div class="stat">\u2694\ufe0f Atk: {c2s['power']} <span class="mult">\u2192 {c2s.get('effective_atk','?')}</span></div>
<div class="stat">\U0001f6e1 Def: {c2s['defense']} <span class="mult">({c2s.get('def_rating','?')}% block)</span></div>
<div class="stat">\u2728 <span class="rarity-{c2s['rarity'].lower().replace(' ','').replace('-','')}">{c2s['rarity']}</span></div>
<div class="stat">\U0001f3ab #{c2s['serial']}</div>
<div class="stat">\u2764\ufe0f {ctx['hp2_start']} HP</div>
<div class="stat">\u2728 Ability rate: {c2s.get('ability_rate','')}%</div>
{ability_html(c2s)}
</div></div></div>
<div class="winner">{'\U0001f3c6 ' + winner_display + ' wins!' if winner_display != 'Tie' else '\U0001f91d Tie!'}</div>
<div class="hp-section">
<div class="hp-row"><span>{n1}: {ctx['hp1_end']}/{ctx['hp1_start']} HP</span>
<div class="hp-bar-bg"><div class="hp-bar {'green' if ctx['hp1_end'] > 0 else 'red'}" id="hp1" style="width:100%"></div></div></div>
<div class="hp-row"><span>{n2}: {ctx['hp2_end']}/{ctx['hp2_start']} HP</span>
<div class="hp-bar-bg"><div class="hp-bar {'green' if ctx['hp2_end'] > 0 else 'red'}" id="hp2" style="width:100%"></div></div></div>
</div>
<div class="legend">
<span>\U0001f4a5 Crit</span><span>\U0001f525 Desperate</span><span>\U0001f4a8 Glancing/Miss</span>
<span>\u2728 Ability</span><span>\U0001f4a4 Stunned</span>
</div>
<div class="log"><h3>\U0001f4dc Battle Log</h3>{log_html}</div>
</div>
<script>
setTimeout(()=>{{document.getElementById('hp1').style.width='{hp1_pct}%';document.getElementById('hp2').style.width='{hp2_pct}%';}},500);
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
    return {"common": "\u26aa", "rare": "\U0001f535",
            "ultrarare": "\U0001f7e3", "legendary": "\U0001f7e1"}.get(key, "\u2728")


# ---------- Telegram handlers ----------
async def cmd_battle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\u2694\ufe0f PFP Battle Bot\n\n"
        "/challenge @username - Start a battle\n"
        "/mystats - View your card stats\n\n"
        "\U0001f4a1 How battles work:\n"
        "\u2022 Power = attack damage\n"
        "\u2022 Defense = % damage blocked\n"
        "\u2022 Rarity & serial give a small edge\n"
        "\u2022 Abilities are read from YOUR card!\n"
        "\u2022 Common cards trigger abilities MORE often\n"
        "\u2022 Crits + comebacks keep it unpredictable\n\n"
        "\U0001f512 Stats hidden until both players upload!\n"
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
        "\U0001f512 Stats hidden until both cards are in!"
    )


async def cmd_mystats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    card = uploaded_cards.get(update.effective_user.id)
    if not card:
        await update.message.reply_text("\u274c Upload a card first!")
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
        f"\U0001f4ca {card['name']}\n\n"
        f"\u2694\ufe0f Power: {card['power']} \u2192 Attack: {atk}\n"
        f"\U0001f6e1 Defense: {card['defense']} \u2192 Block: {int(def_r * 100)}%\n"
        f"{r_emj} {card['rarity']}\n"
        f"\U0001f3ab #{card['serial']}\n"
        f"\u2764\ufe0f HP: {hp}\n"
        f"\u2728 Ability rate: {int(ab_ch * 100)}%/round\n\n"
        f"\U0001f4ac \"{card.get('description', '')}\"\n\n"
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
            await update.message.reply_text("\u26a0\ufe0f Empty file.")
            return

        log.info(f"Downloaded {len(file_bytes)} bytes for @{username}")

        save_path = f"cards/{username}.png"
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        msg = await update.message.reply_text("\U0001f916 Analyzing card...")
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
                f"\u2705 {card['name']} is locked in for @{username}!\n\n"
                f"\U0001f512 Stats hidden until opponent uploads.\n"
                f"\u23f3 Waiting..."
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
                f"\u2705 {card['name']} ready!\n"
                f"Owner: @{username}\n\n"
                f"\u2694\ufe0f Atk: {card['power']} \u2192 {atk}\n"
                f"\U0001f6e1 Block: {int(def_r * 100)}%\n"
                f"{r_emj} {card['rarity']} | \U0001f3ab #{card['serial']}\n"
                f"\u2764\ufe0f HP: {hp}\n"
                f"\u2728 Ability rate: {int(ab_ch * 100)}%\n"
                f"\U0001f4a0 Moves: {ab_names}\n\n"
                f"\U0001f4ac \"{card['description']}\"\n\n"
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
            await msg.edit_text(f"\u2705 {card['name']} locked in!\n\u23f3 Waiting...")
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

        # Build stats for storage/display
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

        # Count events
        abilities_used = [e for e in log_data if e.get("event") == "ability"]
        crits = sum(1 for e in log_data if e.get("event") in ("critical", "desperate"))
        desperate = sum(1 for e in log_data if e.get("event") == "desperate")
        num_rounds = len(set(e["round"] for e in log_data))

        url = f"{RENDER_EXTERNAL_URL}/battle/{bid}"
        kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("\U0001f3ac View Full Replay", url=url)]]
        )

        c1_emj = rarity_emoji(c1["rarity"])
        c2_emj = rarity_emoji(c2["rarity"])
        c1_ab_rate = int(get_ability_trigger_chance(c1["rarity"]) * 100)
        c2_ab_rate = int(get_ability_trigger_chance(c2["rarity"]) * 100)

        result = f"\u2694\ufe0f {c1['name']} vs {c2['name']}\n\n"
        result += (
            f"{c1_emj} {c1['name']} (@{c1['username']})\n"
            f"Atk:{calculate_attack(c1)} Block:{int(calculate_defense_rating(c1)*100)}% "
            f"Ability:{c1_ab_rate}%\n"
            f"\u2764\ufe0f {hp1_end}/{hp1_start} HP\n\n"
        )
        result += (
            f"{c2_emj} {c2['name']} (@{c2['username']})\n"
            f"Atk:{calculate_attack(c2)} Block:{int(calculate_defense_rating(c2)*100)}% "
            f"Ability:{c2_ab_rate}%\n"
            f"\u2764\ufe0f {hp2_end}/{hp2_start} HP\n\n"
        )

        if winner_username:
            result += f"\U0001f3c6 {winner_char} (@{winner_username}) wins!\n"
        else:
            result += "\U0001f91d Tie!\n"

        result += f"\u23f1 {num_rounds} rounds"
        if crits > 0:
            result += f" | \U0001f4a5 {crits} crits"
        if desperate > 0:
            result += f" | \U0001f525 {desperate} comebacks"
        if abilities_used:
            ab_names = list(set(
                e["ability"]["name"] for e in abilities_used if e.get("ability")
            ))
            if ab_names:
                result += f"\n\u2728 Moves used: {', '.join(ab_names[:6])}"

        await msg.edit_text(f"\u2705 {card['name']} locked in! Battle starting...")
        await update.message.reply_text(result, reply_markup=kb)
        log.info(f"=== BATTLE DONE: winner={winner_username or 'Tie'} ===")

        uploaded_cards.pop(cid, None)
        uploaded_cards.pop(oid, None)
        pending_challenges.pop(cid, None)

    except Exception as e:
        log.exception(f"!!! ERROR for @{username}: {e}")
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
