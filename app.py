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

# ---------- Special Abilities ----------
# Each rarity tier has access to different ability pools
# Higher rarity = more powerful abilities + access to lower tier abilities

COMMON_ABILITIES = [
    {"name": "Quick Strike", "emoji": "\u26a1", "type": "attack",
     "desc": "A swift jab", "damage_mult": 1.3, "defense_bypass": 0.0},
    {"name": "Brace", "emoji": "\U0001f6e1", "type": "defense",
     "desc": "Braces for impact", "heal_pct": 0.05, "block_bonus": 0.2},
    {"name": "Focus", "emoji": "\U0001f3af", "type": "buff",
     "desc": "Focuses energy", "next_attack_mult": 1.5},
    {"name": "Taunt", "emoji": "\U0001f4e2", "type": "debuff",
     "desc": "Taunts the enemy", "enemy_miss_chance": 0.3},
]

RARE_ABILITIES = [
    {"name": "Power Surge", "emoji": "\U0001f4a5", "type": "attack",
     "desc": "Unleashes stored energy", "damage_mult": 1.8, "defense_bypass": 0.2},
    {"name": "Iron Wall", "emoji": "\U0001f9f1", "type": "defense",
     "desc": "Becomes nearly invulnerable", "heal_pct": 0.08, "block_bonus": 0.4},
    {"name": "Drain Strike", "emoji": "\U0001f9db", "type": "attack",
     "desc": "Steals life force", "damage_mult": 1.3, "lifesteal_pct": 0.5},
    {"name": "Counter Stance", "emoji": "\U0001f500", "type": "counter",
     "desc": "Prepares to counter-attack", "counter_mult": 2.0},
]

ULTRA_RARE_ABILITIES = [
    {"name": "Void Blast", "emoji": "\U0001f30c", "type": "attack",
     "desc": "Tears a hole in reality", "damage_mult": 2.5, "defense_bypass": 0.5},
    {"name": "Time Warp", "emoji": "\u231b", "type": "special",
     "desc": "Bends time itself", "extra_turns": 1},
    {"name": "Phoenix Shield", "emoji": "\U0001f985", "type": "defense",
     "desc": "Wreathed in protective flames", "heal_pct": 0.15, "block_bonus": 0.5},
    {"name": "Soul Rend", "emoji": "\U0001f480", "type": "attack",
     "desc": "Attacks the very soul", "damage_mult": 2.0, "defense_bypass": 0.75},
]

LEGENDARY_ABILITIES = [
    {"name": "Apocalypse", "emoji": "\u2604\ufe0f", "type": "attack",
     "desc": "Brings forth the end times", "damage_mult": 3.5, "defense_bypass": 0.6},
    {"name": "Divine Resurrection", "emoji": "\U0001f31f", "type": "heal",
     "desc": "Returns from the brink of death", "heal_pct": 0.35},
    {"name": "Omnislash", "emoji": "\u2694\ufe0f", "type": "attack",
     "desc": "Strikes from every direction", "damage_mult": 2.0, "hits": 3, "defense_bypass": 0.3},
    {"name": "Absolute Zero", "emoji": "\u2744\ufe0f", "type": "special",
     "desc": "Freezes time and space", "stun_rounds": 2, "damage_mult": 1.5},
    {"name": "Realm Shatter", "emoji": "\U0001f4a0", "type": "attack",
     "desc": "Shatters the battlefield itself", "damage_mult": 4.0, "defense_bypass": 0.9, "self_damage_pct": 0.1},
]


def get_ability_pool(rarity: str) -> list:
    """Higher rarity gets access to better abilities plus all lower tier ones"""
    key = rarity.lower().replace(" ", "").replace("-", "")
    if key == "legendary":
        return LEGENDARY_ABILITIES + ULTRA_RARE_ABILITIES + RARE_ABILITIES + COMMON_ABILITIES
    elif key == "ultrarare":
        return ULTRA_RARE_ABILITIES + RARE_ABILITIES + COMMON_ABILITIES
    elif key == "rare":
        return RARE_ABILITIES + COMMON_ABILITIES
    else:
        return COMMON_ABILITIES


def get_ability_trigger_chance(rarity: str) -> float:
    """How often abilities trigger per round"""
    key = rarity.lower().replace(" ", "").replace("-", "")
    return {
        "common": 0.08,      # 8% per round
        "rare": 0.12,        # 12% per round
        "ultrarare": 0.18,   # 18% per round
        "legendary": 0.25,   # 25% per round
    }.get(key, 0.08)


# ---------- Claude Vision ----------
claude_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


async def analyze_card_with_claude(file_bytes: bytes) -> dict:
    """Use Claude Vision to extract character name, stats, and flavor from card"""
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
            max_tokens=700,
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
                            "text": """Extract ALL information from this PFP battle card.

Look for:
- Character Name (the name/title of the character on the card)
- Power (attack stat) - read the EXACT number
- Defense (defense stat) - read the EXACT number
- Rarity (Common, Rare, Ultra-Rare, or Legendary)
- Serial Number - labeled "Serial", "S/N", "#", or "422/1999". Use the first number before any slash.
- Description/Flavor text (any lore, backstory, or ability description on the card)

Return ONLY valid JSON (no markdown, no code blocks):
{"name": "<character name>", "power": <number>, "defense": <number>, "rarity": "<rarity>", "serial": <number>, "description": "<any flavor text or description>"}

IMPORTANT: 
- Read EXACT values from the card. Do NOT cap or limit numbers.
- For the name, use the character's actual name shown on the card.
- If no name is visible, describe what the character looks like in 2-3 words.
- For description, include any text about abilities, lore, or backstory.
- If no description, write a brief one based on what you see.

Defaults only if truly not visible: power=50, defense=50, rarity="Common", serial=1000"""
                        }
                    ],
                }
            ],
        )

        response_text = message.content[0].text.strip()
        log.info(f"Claude response: {response_text[:300]}")

        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()

        stats = json.loads(json_text)

        name = str(stats.get("name", "Unknown Warrior")).strip()
        if not name or name.lower() == "unknown":
            name = "Unknown Warrior"

        power = max(1, int(stats.get("power", 50)))
        defense = max(1, int(stats.get("defense", 50)))
        rarity = str(stats.get("rarity", "Common"))
        serial = max(1, int(stats.get("serial", 1000)))
        description = str(stats.get("description", "A mysterious fighter.")).strip()
        if not description:
            description = "A mysterious fighter."

        log.info(f"Extracted: name={name}, power={power}, defense={defense}, rarity={rarity}, serial={serial}")

        return {
            "name": name,
            "power": power,
            "defense": defense,
            "rarity": rarity,
            "serial": serial,
            "description": description,
        }

    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Claude JSON: {e}")
        return {"name": "Unknown Warrior", "power": 50, "defense": 50, "rarity": "Common", "serial": 1000, "description": "A mysterious fighter."}
    except anthropic.APIError as e:
        log.error(f"Anthropic API error: {e}")
        return {"name": "Unknown Warrior", "power": 50, "defense": 50, "rarity": "Common", "serial": 1000, "description": "A mysterious fighter."}
    except Exception as e:
        log.exception(f"Claude API error: {e}")
        return {"name": "Unknown Warrior", "power": 50, "defense": 50, "rarity": "Common", "serial": 1000, "description": "A mysterious fighter."}


# ---------- Stat calculations ----------
def get_rarity_multiplier(rarity: str) -> float:
    key = rarity.lower().replace(" ", "").replace("-", "")
    return RARITY_MULTIPLIER.get(key, 1.0)


def get_serial_multiplier(serial: int) -> float:
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
    """HP = defense-weighted base * rarity * serial"""
    defense = card.get("defense", 50)
    power = card.get("power", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))

    base_hp = (defense * 0.7) + (power * 0.3)
    hp = int(base_hp * 3.0 * rarity_mult * serial_mult)
    return max(10, hp)


def calculate_attack(card: dict) -> int:
    """Attack = power * rarity * serial"""
    power = card.get("power", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))

    attack = int(power * rarity_mult * serial_mult)
    return max(1, attack)


def calculate_defense_rating(card: dict) -> float:
    """Defense rating = % damage blocked (5% to 75%)"""
    defense = card.get("defense", 50)
    rarity_mult = get_rarity_multiplier(card.get("rarity", "Common"))
    serial_mult = get_serial_multiplier(int(card.get("serial", 1000)))

    effective_def = defense * rarity_mult * serial_mult
    reduction = effective_def / (effective_def + 200.0)
    reduction = min(0.75, max(0.05, reduction))
    return round(reduction, 3)


# ---------- Battle simulation ----------
def simulate_battle(card1: dict, card2: dict):
    """
    Full battle sim with:
    - Power vs Defense: attack damage reduced by defender's block %
    - Rarity & serial multiply everything
    - Critical hits (15% base, up to 40% when losing)
    - Glancing blows (10%)
    - Special abilities triggered by rarity
    - Desperate comeback mechanic
    """
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

    ability_pool1 = get_ability_pool(card1.get("rarity", "Common"))
    ability_pool2 = get_ability_pool(card2.get("rarity", "Common"))
    ability_chance1 = get_ability_trigger_chance(card1.get("rarity", "Common"))
    ability_chance2 = get_ability_trigger_chance(card2.get("rarity", "Common"))

    battle_log = []
    round_num = 0

    # Damage scaling for ~10-20 round battles
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

    # Track buffs/debuffs
    p1_next_mult = 1.0
    p2_next_mult = 1.0
    p1_stun = 0
    p2_stun = 0
    p1_counter = False
    p2_counter = False
    p1_miss_chance = 0.0
    p2_miss_chance = 0.0

    while hp1 > 0 and hp2 > 0 and round_num < 50:
        round_num += 1

        # Reset per-round debuffs
        if round_num > 1:
            p1_miss_chance = max(0, p1_miss_chance - 0.15)
            p2_miss_chance = max(0, p2_miss_chance - 0.15)

        # --- Player 1 turn ---
        if p1_stun > 0:
            p1_stun -= 1
            battle_log.append({
                "round": round_num, "attacker": 1, "damage": 0,
                "event": "stunned", "ability": None,
                "hp1": max(0, hp1), "hp2": max(0, hp2),
                "text": f"{name1} is stunned and can't move!"
            })
        else:
            # Check for ability trigger
            ability_used = None
            if random.random() < ability_chance1:
                ability_used = random.choice(ability_pool1)
                hp1, hp2, p1_next_mult, p2_stun, p1_counter, p2_miss_chance, ability_log = execute_ability(
                    ability_used, atk1, def2, damage_scale, hp1, hp2,
                    hp1_start, hp2_start, name1, name2, p1_next_mult
                )
                battle_log.append({
                    "round": round_num, "attacker": 1,
                    "damage": ability_log.get("damage", 0),
                    "event": "ability", "ability": ability_used,
                    "hp1": max(0, hp1), "hp2": max(0, hp2),
                    "text": ability_log["text"]
                })
                if hp2 <= 0:
                    break
            else:
                # Normal attack
                if random.random() < p1_miss_chance:
                    battle_log.append({
                        "round": round_num, "attacker": 1, "damage": 0,
                        "event": "miss", "ability": None,
                        "hp1": max(0, hp1), "hp2": max(0, hp2),
                        "text": f"{name1}'s attack misses!"
                    })
                else:
                    dmg, event = calculate_round_damage(
                        atk1, def2, damage_scale,
                        hp1, hp2, hp1_start, hp2_start, p1_next_mult
                    )
                    p1_next_mult = 1.0  # Reset after use

                    # Counter check
                    if p2_counter:
                        counter_dmg = max(1, int(dmg * 0.5))
                        hp1 -= counter_dmg
                        p2_counter = False
                        battle_log.append({
                            "round": round_num, "attacker": 1, "damage": dmg,
                            "event": "countered", "ability": None,
                            "hp1": max(0, hp1), "hp2": max(0, hp2),
                            "text": f"{name1} deals {dmg} but {name2} counters for {counter_dmg}!"
                        })
                        hp2 -= dmg
                    else:
                        hp2 -= dmg
                        battle_log.append({
                            "round": round_num, "attacker": 1, "damage": dmg,
                            "event": event, "ability": None,
                            "hp1": max(0, hp1), "hp2": max(0, hp2),
                            "text": get_attack_text(name1, name2, dmg, event)
                        })

                if hp2 <= 0:
                    break

        # --- Player 2 turn ---
        if p2_stun > 0:
            p2_stun -= 1
            battle_log.append({
                "round": round_num, "attacker": 2, "damage": 0,
                "event": "stunned", "ability": None,
                "hp1": max(0, hp1), "hp2": max(0, hp2),
                "text": f"{name2} is stunned and can't move!"
            })
        else:
            ability_used = None
            if random.random() < ability_chance2:
                ability_used = random.choice(ability_pool2)
                hp2, hp1, p2_next_mult, p1_stun, p2_counter, p1_miss_chance, ability_log = execute_ability(
                    ability_used, atk2, def1, damage_scale, hp2, hp1,
                    hp2_start, hp1_start, name2, name1, p2_next_mult
                )
                battle_log.append({
                    "round": round_num, "attacker": 2,
                    "damage": ability_log.get("damage", 0),
                    "event": "ability", "ability": ability_used,
                    "hp1": max(0, hp1), "hp2": max(0, hp2),
                    "text": ability_log["text"]
                })
                if hp1 <= 0:
                    break
            else:
                if random.random() < p2_miss_chance:
                    battle_log.append({
                        "round": round_num, "attacker": 2, "damage": 0,
                        "event": "miss", "ability": None,
                        "hp1": max(0, hp1), "hp2": max(0, hp2),
                        "text": f"{name2}'s attack misses!"
                    })
                else:
                    dmg, event = calculate_round_damage(
                        atk2, def1, damage_scale,
                        hp2, hp1, hp2_start, hp1_start, p2_next_mult
                    )
                    p2_next_mult = 1.0

                    if p1_counter:
                        counter_dmg = max(1, int(dmg * 0.5))
                        hp2 -= counter_dmg
                        p1_counter = False
                        battle_log.append({
                            "round": round_num, "attacker": 2, "damage": dmg,
                            "event": "countered", "ability": None,
                            "hp1": max(0, hp1), "hp2": max(0, hp2),
                            "text": f"{name2} deals {dmg} but {name1} counters for {counter_dmg}!"
                        })
                        hp1 -= dmg
                    else:
                        hp1 -= dmg
                        battle_log.append({
                            "round": round_num, "attacker": 2, "damage": dmg,
                            "event": event, "ability": None,
                            "hp1": max(0, hp1), "hp2": max(0, hp2),
                            "text": get_attack_text(name2, name1, dmg, event)
                        })

    return max(0, hp1), max(0, hp2), hp1_start, hp2_start, battle_log


def execute_ability(ability, attacker_atk, defender_def, damage_scale,
                    attacker_hp, defender_hp, attacker_max, defender_max,
                    attacker_name, defender_name, current_mult):
    """Execute a special ability and return updated state"""
    ab_type = ability["type"]
    ab_name = ability["name"]
    ab_emoji = ability["emoji"]
    damage_dealt = 0
    next_mult = current_mult
    stun_enemy = 0
    set_counter = False
    enemy_miss = 0.0

    if ab_type == "attack":
        dmg_mult = ability.get("damage_mult", 1.5)
        def_bypass = ability.get("defense_bypass", 0.0)
        hits = ability.get("hits", 1)
        lifesteal = ability.get("lifesteal_pct", 0.0)
        self_dmg = ability.get("self_damage_pct", 0.0)

        effective_def = defender_def * (1.0 - def_bypass)
        total_dmg = 0

        for _ in range(hits):
            base = attacker_atk * damage_scale * random.uniform(0.8, 1.2) * dmg_mult
            hit_dmg = max(1, int(base * (1.0 - effective_def)))
            total_dmg += hit_dmg

        defender_hp -= total_dmg
        damage_dealt = total_dmg

        if lifesteal > 0:
            heal = int(total_dmg * lifesteal)
            attacker_hp = min(attacker_max, attacker_hp + heal)
            text = f"{ab_emoji} {attacker_name} uses {ab_name}! Deals {total_dmg} and heals {heal}!"
        elif self_dmg > 0:
            self_hit = int(attacker_max * self_dmg)
            attacker_hp -= self_hit
            text = f"{ab_emoji} {attacker_name} uses {ab_name}! Deals {total_dmg} but takes {self_hit} recoil!"
        elif hits > 1:
            text = f"{ab_emoji} {attacker_name} uses {ab_name}! {hits} hits for {total_dmg} total damage!"
        else:
            text = f"{ab_emoji} {attacker_name} uses {ab_name}! Deals {total_dmg} damage!"

    elif ab_type == "defense":
        heal_pct = ability.get("heal_pct", 0.05)
        heal = int(attacker_max * heal_pct)
        attacker_hp = min(attacker_max, attacker_hp + heal)
        text = f"{ab_emoji} {attacker_name} uses {ab_name}! Heals {heal} HP!"

    elif ab_type == "heal":
        heal_pct = ability.get("heal_pct", 0.2)
        heal = int(attacker_max * heal_pct)
        attacker_hp = min(attacker_max, attacker_hp + heal)
        text = f"{ab_emoji} {attacker_name} uses {ab_name}! Restores {heal} HP!"

    elif ab_type == "buff":
        next_mult = ability.get("next_attack_mult", 1.5)
        text = f"{ab_emoji} {attacker_name} uses {ab_name}! Next attack powered up {next_mult}x!"

    elif ab_type == "debuff":
        enemy_miss = ability.get("enemy_miss_chance", 0.3)
        text = f"{ab_emoji} {attacker_name} uses {ab_name}! {defender_name} is distracted!"

    elif ab_type == "counter":
        set_counter = True
        text = f"{ab_emoji} {attacker_name} uses {ab_name}! Ready to counter-attack!"

    elif ab_type == "special":
        extra = ability.get("extra_turns", 0)
        stun = ability.get("stun_rounds", 0)
        dmg_mult = ability.get("damage_mult", 1.0)

        if stun > 0:
            stun_enemy = stun
            if dmg_mult > 1.0:
                base = attacker_atk * damage_scale * random.uniform(0.8, 1.2) * dmg_mult
                hit_dmg = max(1, int(base * (1.0 - defender_def * 0.5)))
                defender_hp -= hit_dmg
                damage_dealt = hit_dmg
                text = f"{ab_emoji} {attacker_name} uses {ab_name}! Deals {hit_dmg} and stuns for {stun} rounds!"
            else:
                text = f"{ab_emoji} {attacker_name} uses {ab_name}! {defender_name} is stunned for {stun} rounds!"
        elif extra > 0:
            base = attacker_atk * damage_scale * random.uniform(0.9, 1.3) * 1.5
            hit_dmg = max(1, int(base * (1.0 - defender_def)))
            defender_hp -= hit_dmg
            damage_dealt = hit_dmg
            text = f"{ab_emoji} {attacker_name} uses {ab_name}! Warps time for {hit_dmg} damage and attacks again!"
        else:
            text = f"{ab_emoji} {attacker_name} uses {ab_name}!"
    else:
        text = f"{ab_emoji} {attacker_name} uses {ab_name}!"

    return (
        attacker_hp, defender_hp, next_mult, stun_enemy,
        set_counter, enemy_miss,
        {"text": text, "damage": damage_dealt}
    )


def calculate_round_damage(attacker_atk, defender_def, damage_scale,
                           attacker_hp, defender_hp, attacker_max, defender_max,
                           attack_mult=1.0):
    """Calculate one normal attack with crits/glancing"""
    base_dmg = attacker_atk * damage_scale * random.uniform(0.7, 1.3) * attack_mult

    attacker_hp_pct = attacker_hp / max(1, attacker_max)
    defender_hp_pct = defender_hp / max(1, defender_max)
    hp_disadvantage = max(0, defender_hp_pct - attacker_hp_pct)

    crit_chance = 0.15 + (hp_disadvantage * 0.5)
    crit_chance = min(0.40, crit_chance)
    glancing_chance = 0.10

    roll = random.random()
    event = "normal"

    if roll < crit_chance:
        event = "desperate" if attacker_hp_pct < 0.3 else "critical"
        effective_def = defender_def * 0.5
        damage_after_def = base_dmg * 2.0 * (1.0 - effective_def)
    elif roll < crit_chance + glancing_chance:
        event = "glancing"
        damage_after_def = base_dmg * 0.5 * (1.0 - defender_def)
    else:
        damage_after_def = base_dmg * (1.0 - defender_def)

    return max(1, int(damage_after_def)), event


def get_attack_text(attacker, defender, damage, event):
    """Generate flavorful attack text"""
    if event == "desperate":
        phrases = [
            f"\U0001f525 {attacker} refuses to fall! Desperate strike for {damage}!",
            f"\U0001f525 {attacker} screams and unleashes {damage} damage!",
            f"\U0001f525 With nothing to lose, {attacker} crits for {damage}!",
        ]
    elif event == "critical":
        phrases = [
            f"\U0001f4a5 {attacker} lands a critical hit for {damage}!",
            f"\U0001f4a5 Perfect strike! {attacker} crits {defender} for {damage}!",
            f"\U0001f4a5 {attacker} finds a weak spot! {damage} damage!",
        ]
    elif event == "glancing":
        phrases = [
            f"\U0001f4a8 {attacker}'s attack glances off for only {damage}.",
            f"\U0001f4a8 {defender} deflects! Only {damage} gets through.",
            f"\U0001f4a8 Weak hit from {attacker}... {damage} damage.",
        ]
    else:
        phrases = [
            f"{attacker} strikes {defender} for {damage} damage.",
            f"{attacker} attacks! {damage} damage to {defender}.",
            f"{attacker} hits {defender} for {damage}.",
        ]
    return random.choice(phrases)


# ---------- Battle HTML ----------
def save_battle_html(battle_id: str, ctx: dict):
    """Generate battle replay HTML with character names and abilities."""
    os.makedirs("battles", exist_ok=True)

    log_html = ""
    for e in ctx.get("battle_log", [])[:40]:
        event = e.get("event", "normal")
        text = e.get("text", "")

        if event == "ability":
            ab = e.get("ability", {})
            style = "border-left:3px solid #e040fb;background:rgba(224,64,251,0.12)"
        elif event == "critical":
            style = "border-left:3px solid #ffd740;background:rgba(255,215,64,0.1)"
        elif event == "desperate":
            style = "border-left:3px solid #ff5722;background:rgba(255,87,34,0.15)"
        elif event == "glancing":
            style = "border-left:3px solid #666;background:rgba(100,100,100,0.1)"
        elif event == "stunned":
            style = "border-left:3px solid #00bcd4;background:rgba(0,188,212,0.1)"
        elif event == "miss":
            style = "border-left:3px solid #999;background:rgba(150,150,150,0.05)"
        elif event == "countered":
            style = "border-left:3px solid #ff9800;background:rgba(255,152,0,0.1)"
        else:
            style = "border-left:3px solid #ff6b6b;background:rgba(255,255,255,0.03)"

        hp_display = f" <span style='color:#666;font-size:0.85em'>[{e['hp1']} vs {e['hp2']}]</span>"
        log_html += f'<div style="{style};padding:6px 8px;margin:3px 0;border-radius:3px">R{e["round"]}: {text}{hp_display}</div>\n'

    hp1_pct = max(0, int(ctx["hp1_end"] / max(1, ctx["hp1_start"]) * 100))
    hp2_pct = max(0, int(ctx["hp2_end"] / max(1, ctx["hp2_start"]) * 100))

    c1s = ctx["card1_stats"]
    c2s = ctx["card2_stats"]
    n1 = ctx["card1_char_name"]
    n2 = ctx["card2_char_name"]

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
.desc{{font-size:0.8em;color:#aaa;font-style:italic;margin:6px 0;padding:6px;background:rgba(0,0,0,0.2);border-radius:5px}}
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
<div class="owner">Owner: @{ctx['card1_name']}</div>
<div class="desc">"{c1s.get('description','')}"</div>
<div class="stats">
<div class="stat">\u2694\ufe0f Atk: {c1s['power']} <span class="mult">\u2192 {c1s.get('effective_atk','?')}</span></div>
<div class="stat">\U0001f6e1 Def: {c1s['defense']} <span class="mult">({c1s.get('def_rating','?')}% block)</span></div>
<div class="stat">\u2728 <span class="rarity-{c1s['rarity'].lower().replace(' ','').replace('-','')}">{c1s['rarity']}</span> <span class="mult">({c1s.get('rarity_mult','?')}x)</span></div>
<div class="stat">\U0001f3ab #{c1s['serial']} <span class="mult">({c1s.get('serial_mult','?')}x)</span></div>
<div class="stat">\u2764\ufe0f HP: {ctx['hp1_start']}</div>
</div></div>
<div class="vs">VS</div>
<div class="fighter">
<div class="char-name">{n2}</div>
<div class="owner">Owner: @{ctx['card2_name']}</div>
<div class="desc">"{c2s.get('description','')}"</div>
<div class="stats">
<div class="stat">\u2694\ufe0f Atk: {c2s['power']} <span class="mult">\u2192 {c2s.get('effective_atk','?')}</span></div>
<div class="stat">\U0001f6e1 Def: {c2s['defense']} <span class="mult">({c2s.get('def_rating','?')}% block)</span></div>
<div class="stat">\u2728 <span class="rarity-{c2s['rarity'].lower().replace(' ','').replace('-','')}">{c2s['rarity']}</span> <span class="mult">({c2s.get('rarity_mult','?')}x)</span></div>
<div class="stat">\U0001f3ab #{c2s['serial']} <span class="mult">({c2s.get('serial_mult','?')}x)</span></div>
<div class="stat">\u2764\ufe0f HP: {ctx['hp2_start']}</div>
</div></div></div>
<div class="winner">{'\U0001f3c6 ' + n1 + ' wins!' if ctx['winner_name'] != 'Tie' and ctx['winner_name'] == ctx['card1_name'] else ('\U0001f3c6 ' + n2 + ' wins!' if ctx['winner_name'] != 'Tie' else '\U0001f91d Tie!')}</div>
<div class="hp-section">
<div class="hp-row">
<span>{n1}: {ctx['hp1_end']}/{ctx['hp1_start']} HP</span>
<div class="hp-bar-bg"><div class="hp-bar {'green' if ctx['hp1_end'] > 0 else 'red'}" id="hp1" style="width:100%"></div></div>
</div>
<div class="hp-row">
<span>{n2}: {ctx['hp2_end']}/{ctx['hp2_start']} HP</span>
<div class="hp-bar-bg"><div class="hp-bar {'green' if ctx['hp2_end'] > 0 else 'red'}" id="hp2" style="width:100%"></div></div>
</div>
</div>
<div class="legend">
<span>\U0001f4a5 Critical</span>
<span>\U0001f525 Desperate</span>
<span>\U0001f4a8 Glancing</span>
<span>\u2728 Ability</span>
<span>\U0001f4a4 Stunned</span>
<span>\U0001f500 Counter</span>
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


# ---------- Display helpers ----------
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
        "\u2022 \u2694\ufe0f Power = attack damage\n"
        "\u2022 \U0001f6e1 Defense = % damage blocked\n"
        "\u2022 \u2728 Rarity multiplies everything + unlocks abilities\n"
        "\u2022 \U0001f3ab Low serial # = huge bonus\n"
        "\u2022 \U0001f4a5 Critical hits when you're losing!\n"
        "\u2022 \u2728 Special abilities trigger randomly!\n"
        "\u2022 Legendary cards get Apocalypse, Omnislash & more\n\n"
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
    ab_chance = get_ability_trigger_chance(card["rarity"])

    await update.message.reply_text(
        f"\U0001f4ca {card['name']}\n\n"
        f"\u2694\ufe0f Power: {card['power']} \u2192 Attack: {atk}\n"
        f"\U0001f6e1 Defense: {card['defense']} \u2192 Blocks {int(def_rating * 100)}%\n"
        f"{r_emj} Rarity: {card['rarity']} ({r_mult}x)\n"
        f"\U0001f3ab Serial: #{card['serial']} ({s_mult}x)\n"
        f"\u2764\ufe0f HP: {hp}\n"
        f"\u2728 Ability chance: {int(ab_chance * 100)}% per round\n\n"
        f"\U0001f4ac \"{card.get('description', 'A mysterious fighter.')}\""
    )


async def debug_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    log.info(f"DEBUG: from {update.effective_user.username} "
             f"photo={bool(update.message.photo)} doc={bool(update.message.document)}")


async def handler_card_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    log.info(f"Card upload! User: {update.effective_user.username}")

    user = update.effective_user
    username = (user.username or f"user{user.id}").lower()
    user_id = user.id

    try:
        file_obj = None
        if update.message.photo:
            log.info("Photo detected")
            file_obj = await update.message.photo[-1].get_file()
        elif update.message.document:
            log.info(f"Document: {update.message.document.mime_type}")
            file_obj = await update.message.document.get_file()
        else:
            return

        file_bytes = await file_obj.download_as_bytearray()
        if len(file_bytes) == 0:
            await update.message.reply_text("\u26a0\ufe0f Empty file. Try again.")
            return

        save_path = f"cards/{username}.png"
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        msg = await update.message.reply_text("\U0001f916 Analyzing card...")

        parsed = await analyze_card_with_claude(bytes(file_bytes))

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
        }

        uploaded_cards[user_id] = card

        # Check challenge status
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

        # HIDE if in challenge waiting for opponent
        if in_challenge and not opponent_ready:
            await msg.edit_text(
                f"\u2705 {card['name']} is locked in for @{username}!\n\n"
                f"\U0001f512 Stats hidden until opponent uploads.\n"
                f"\u23f3 Waiting..."
            )
            return

        # NOT in challenge - show full stats
        if not in_challenge:
            hp = calculate_hp(card)
            atk = calculate_attack(card)
            def_r = calculate_defense_rating(card)
            r_mult = get_rarity_multiplier(card["rarity"])
            s_mult = get_serial_multiplier(card["serial"])
            r_emj = rarity_emoji(card["rarity"])
            ab_chance = get_ability_trigger_chance(card["rarity"])

            await msg.edit_text(
                f"\u2705 {card['name']} ready!\n"
                f"Owner: @{username}\n\n"
                f"\u2694\ufe0f Power: {card['power']} \u2192 Atk: {atk}\n"
                f"\U0001f6e1 Defense: {card['defense']} \u2192 Block: {int(def_r * 100)}%\n"
                f"{r_emj} {card['rarity']} ({r_mult}x)\n"
                f"\U0001f3ab #{card['serial']} ({s_mult}x)\n"
                f"\u2764\ufe0f HP: {hp}\n"
                f"\u2728 Ability: {int(ab_chance * 100)}%/round\n\n"
                f"\U0001f4ac \"{card['description']}\"\n\n"
                f"Use /challenge @username to battle!"
            )
            return

        # BOTH READY - battle time
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
            await msg.edit_text(f"\u2705 {card['name']} locked in!\n\u23f3 Waiting...")
            return

        cid, oid = triggered_pair
        c1, c2 = uploaded_cards[cid], uploaded_cards[oid]

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

        c1_r = get_rarity_multiplier(c1["rarity"])
        c1_s = get_serial_multiplier(c1["serial"])
        c2_r = get_rarity_multiplier(c2["rarity"])
        c2_s = get_serial_multiplier(c2["serial"])

        bid = str(uuid.uuid4())
        battle_ctx = {
            "card1_name": c1["username"],
            "card2_name": c2["username"],
            "card1_char_name": c1["name"],
            "card2_char_name": c2["name"],
            "card1_stats": {
                "power": c1["power"], "defense": c1["defense"],
                "rarity": c1["rarity"], "serial": c1["serial"],
                "rarity_mult": c1_r, "serial_mult": c1_s,
                "effective_atk": calculate_attack(c1),
                "def_rating": int(calculate_defense_rating(c1) * 100),
                "description": c1.get("description", ""),
            },
            "card2_stats": {
                "power": c2["power"], "defense": c2["defense"],
                "rarity": c2["rarity"], "serial": c2["serial"],
                "rarity_mult": c2_r, "serial_mult": c2_s,
                "effective_atk": calculate_attack(c2),
                "def_rating": int(calculate_defense_rating(c2) * 100),
                "description": c2.get("description", ""),
            },
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

        result = f"\u2694\ufe0f {c1['name']} vs {c2['name']}\n\n"

        result += (
            f"{c1_emj} {c1['name']} (@{c1['username']})\n"
            f"Atk:{calculate_attack(c1)} Block:{int(calculate_defense_rating(c1)*100)}%\n"
            f"\u2764\ufe0f {hp1_end}/{hp1_start} HP\n\n"
        )
        result += (
            f"{c2_emj} {c2['name']} (@{c2['username']})\n"
            f"Atk:{calculate_attack(c2)} Block:{int(calculate_defense_rating(c2)*100)}%\n"
            f"\u2764\ufe0f {hp2_end}/{hp2_start} HP\n\n"
        )

        if winner_username:
            result += f"\U0001f3c6 {winner_char} (@{winner_username}) wins!\n"
        else:
            result += "\U0001f91d It's a Tie!\n"

        result += f"\u23f1 {num_rounds} rounds"
        if crits > 0:
            result += f" | \U0001f4a5 {crits} crits"
        if desperate > 0:
            result += f" | \U0001f525 {desperate} comebacks"
        if abilities_used:
            ab_names = list(set(e["ability"]["name"] for e in abilities_used if e.get("ability")))
            if ab_names:
                result += f"\n\u2728 Abilities: {', '.join(ab_names[:4])}"

        await msg.edit_text(f"\u2705 {card['name']} locked in! Battle starting...")
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
