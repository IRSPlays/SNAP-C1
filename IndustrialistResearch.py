"""
============================================================
  The Industrialist — Research Center 2 (Tier 2) Calculator
  Real-time production chain for Crankshaft → RS2
============================================================

Goal: Keep one Research Station 2 fed at exactly 0.5 crankshafts/sec
      (the RS2 consumption rate) without over- or under-producing.

Crankshaft RP multiplier: x35  |  Sell value: $754
"""

# ──────────────────────────────────────────────────────────────
# 1.  MACHINE DATA (times in seconds, quantities per cycle)
# ──────────────────────────────────────────────────────────────

MACHINES = {
    # ── Mining & Smelting ──────────────────────────────────────
    "Iron Drill": {
        "inputs":  {},
        "outputs": {"Raw Iron": 1},
        "time_s":  15,
        "power":   "—",
    },
    "Electric Furnace": {
        "inputs":  {"Raw Iron": 1},
        "outputs": {"Liquid Iron": 1},     # approximate 1:1
        "time_s":  5,
        "power":   "3 kMF/s",
    },
    "Ingot Molder": {
        "inputs":  {"Liquid Iron": 1.8},
        "outputs": {"Iron Ingot": 1},
        "time_s":  10,
        "power":   "—",
    },

    # ── Steel Production ──────────────────────────────────────
    "Blast Furnace (Steel)": {
        "inputs":  {"Iron Ingot": 1, "Coal": 4},
        "outputs": {"Steel Ingot": 2},
        "time_s":  5,
        "power":   "10 kMF/s",
    },

    # ── Steel Rods (via Roller) ───────────────────────────────
    "Roller (Steel Rod)": {
        "inputs":  {"Steel Ingot": 2},
        "outputs": {"Steel Rod": 1},
        "time_s":  8,
        "power":   "6 kMF/s",
    },

    # ── Gears (Steel Ingot → Press → Steel Plate → Roller → Gear)
    "Press (Steel Plate)": {
        "inputs":  {"Steel Ingot": 1},
        "outputs": {"Steel Plate": 1},
        "time_s":  4,
        "power":   "12.5 kMF/s",
    },
    "Roller (Gear)": {
        "inputs":  {"Steel Plate": 1},
        "outputs": {"Gear": 1},
        "time_s":  8,
        "power":   "6 kMF/s",
    },

    # ── Crankshaft Assembly ───────────────────────────────────
    "Craft Assembler (Crankshaft)": {
        "inputs":  {"Steel Rod": 2, "Gear": 4},
        "outputs": {"Crankshaft": 1},
        "time_s":  15,
        "power":   "4 kMF/s",
    },

    # ── Research Station 2 ────────────────────────────────────
    "Research Station 2": {
        "inputs":  {"Crankshaft": 0.5},    # 0.5 items/sec continuous
        "outputs": {"RP": "x35 per item"},
        "time_s":  1,                       # continuous per second
        "power":   "requires power",
    },
}


# ──────────────────────────────────────────────────────────────
# 2.  DEMAND-DRIVEN CALCULATIONS
#     Target: 0.5 crankshafts per second (1 RS2)
# ──────────────────────────────────────────────────────────────

import math

TARGET_CRANKSHAFTS_PER_SEC = 0.5   # RS2 consumption rate

def items_per_sec(quantity_per_cycle, cycle_time):
    """Output rate of one machine."""
    return quantity_per_cycle / cycle_time

def machines_needed(demand_per_sec, supply_per_machine_per_sec):
    """How many machines to meet demand (round up)."""
    return math.ceil(demand_per_sec / supply_per_machine_per_sec)


# ── Crankshaft Assemblers ─────────────────────────────────
crank_rate   = items_per_sec(1, 15)                          # 1 / 15 ≈ 0.0667/s
crank_assemblers = machines_needed(TARGET_CRANKSHAFTS_PER_SEC, crank_rate)  # 8

# ── Steel Rod demand ──────────────────────────────────────
steel_rod_demand = TARGET_CRANKSHAFTS_PER_SEC * 2            # 1.0 rod/s
rod_rate         = items_per_sec(1, 8)                       # 0.125/s
rollers_rod      = machines_needed(steel_rod_demand, rod_rate)  # 8

# ── Gear demand ───────────────────────────────────────────
gear_demand      = TARGET_CRANKSHAFTS_PER_SEC * 4            # 2.0 gears/s
gear_rate        = items_per_sec(1, 8)                       # 0.125/s
rollers_gear     = machines_needed(gear_demand, gear_rate)   # 16

# ── Steel Plate demand (for Gears) ───────────────────────
plate_demand     = gear_demand * 1                           # 2.0 plates/s
plate_rate       = items_per_sec(1, 4)                       # 0.25/s
presses          = machines_needed(plate_demand, plate_rate) # 8

# ── Total Steel Ingot demand ─────────────────────────────
#    Steel Rods: 2 ingots per rod   → 1.0 rod/s × 2 = 2.0 ingots/s
#    Steel Plates: 1 ingot per plate → 2.0 plates/s × 1 = 2.0 ingots/s
#    Total = 4.0 ingots/s
steel_ingot_demand = (steel_rod_demand * 2) + (plate_demand * 1)
blast_rate         = items_per_sec(2, 5)                     # 0.4/s
blast_furnaces     = machines_needed(steel_ingot_demand, blast_rate)  # 10

# ── Iron Ingot demand (for Blast Furnace) ────────────────
iron_ingot_demand  = steel_ingot_demand / 2 * 1              # 1 iron ingot per 2 steel → 2.0/s
iron_ingot_demand  = (steel_ingot_demand / 2)                # = 2.0 iron ingots/s
molder_rate        = items_per_sec(1, 10)                    # 0.1/s
molders            = machines_needed(iron_ingot_demand, molder_rate)  # 20

# ── Liquid Iron demand ───────────────────────────────────
liquid_iron_demand = iron_ingot_demand * 1.8                 # 1.8 per iron ingot → 3.6/s
furnace_rate       = items_per_sec(1, 5)                     # 0.2/s
elec_furnaces      = machines_needed(liquid_iron_demand, furnace_rate)  # 18

# ── Raw Iron demand (mining) ─────────────────────────────
raw_iron_demand    = liquid_iron_demand * 1                  # 3.6/s
drill_rate         = items_per_sec(1, 15)                    # 0.0667/s
iron_drills        = machines_needed(raw_iron_demand, drill_rate)  # 54

# ── Coal demand (for Blast Furnace) ──────────────────────
coal_demand        = (steel_ingot_demand / 2) * 4            # 4 coal per blast cycle (produces 2 steel) → 8.0/s
coal_drill_rate    = items_per_sec(1, 15)                    # assuming same drill rate
coal_drills        = machines_needed(coal_demand, coal_drill_rate)  # 120


# ──────────────────────────────────────────────────────────────
# 3.  PRINT FULL PRODUCTION LAYOUT
# ──────────────────────────────────────────────────────────────

def print_header(title):
    w = 62
    print("═" * w)
    print(f"  {title}")
    print("═" * w)

def print_row(machine, count, rate_each, total_rate, notes=""):
    print(f"  {machine:<35} ×{count:>3}   │ {total_rate:>6.2f}/s  {notes}")

def main():
    print()
    print_header("THE INDUSTRIALIST — RS2 CRANKSHAFT PRODUCTION CHAIN")
    print(f"  Target: {TARGET_CRANKSHAFTS_PER_SEC} crankshafts/sec (1× Research Station 2)")
    print(f"  Crankshaft RP: ×35 multiplier  |  Value: $754")
    print(f"  RP generation: ~{TARGET_CRANKSHAFTS_PER_SEC * 35:.1f} RP/sec")
    print()

    print("─" * 62)
    print("  STAGE                               QTY  │ THROUGHPUT")
    print("─" * 62)

    print()
    print("  ▸ MINING")
    print_row("Iron Drill", iron_drills, drill_rate, iron_drills * drill_rate, "(raw iron)")
    print_row("Coal Drill", coal_drills, coal_drill_rate, coal_drills * coal_drill_rate, "(coal)")

    print()
    print("  ▸ SMELTING & INGOTS")
    print_row("Electric Furnace", elec_furnaces, furnace_rate, elec_furnaces * furnace_rate, "(liquid iron)")
    print_row("Ingot Molder", molders, molder_rate, molders * molder_rate, "(iron ingots)")
    print_row("Blast Furnace", blast_furnaces, blast_rate, blast_furnaces * blast_rate, "(steel ingots)")

    print()
    print("  ▸ STEEL PROCESSING")
    print_row("Roller (Steel Rod)", rollers_rod, rod_rate, rollers_rod * rod_rate, "(steel rods)")
    print_row("Press (Steel Plate)", presses, plate_rate, presses * plate_rate, "(for gears)")
    print_row("Roller (Gear)", rollers_gear, gear_rate, rollers_gear * gear_rate, "(gears)")

    print()
    print("  ▸ ASSEMBLY")
    print_row("Craft Assembler (Crankshaft)", crank_assemblers, crank_rate, crank_assemblers * crank_rate, "(crankshafts)")

    print()
    print("  ▸ RESEARCH")
    print_row("Research Station 2", 1, TARGET_CRANKSHAFTS_PER_SEC, TARGET_CRANKSHAFTS_PER_SEC, "(consuming)")

    total_machines = (iron_drills + coal_drills + elec_furnaces + molders +
                      blast_furnaces + rollers_rod + presses + rollers_gear +
                      crank_assemblers + 1)

    print()
    print("─" * 62)
    print(f"  TOTAL MACHINES: {total_machines}")
    print("─" * 62)

    # ── Summary Table ─────────────────────────────────────────
    print()
    print_header("QUICK REFERENCE — MACHINE COUNTS")
    summary = [
        ("Iron Drill",                  iron_drills),
        ("Coal Drill",                  coal_drills),
        ("Electric Furnace",            elec_furnaces),
        ("Ingot Molder",                molders),
        ("Blast Furnace",               blast_furnaces),
        ("Roller (Steel Rod)",          rollers_rod),
        ("Press (Steel Plate)",         presses),
        ("Roller (Gear)",               rollers_gear),
        ("Craft Assembler (Crankshaft)", crank_assemblers),
        ("Research Station 2",          1),
    ]
    for name, count in summary:
        bar = "█" * count + "░" * max(0, 60 - count)
        print(f"  {name:<35} ×{count:>3}  {bar[:40]}")

    print()
    print("═" * 62)
    print("  TIP: If drills are too many, buy raw iron/coal from the")
    print("  market or use higher-tier drills (Steel Drill = 10s cycle).")
    print("  An Advanced Assembler crafts crankshafts in 5s (not 15s),")
    print("  reducing assemblers from 8 → 3.")
    print("═" * 62)

    # ── Optimized variant with Advanced Assembler ─────────────
    print()
    print_header("OPTIMIZED VARIANT — ADVANCED ASSEMBLER (5s craft)")
    adv_crank_rate = items_per_sec(1, 5)
    adv_assemblers = machines_needed(TARGET_CRANKSHAFTS_PER_SEC, adv_crank_rate)
    opt_total = (iron_drills + coal_drills + elec_furnaces + molders +
                 blast_furnaces + rollers_rod + presses + rollers_gear +
                 adv_assemblers + 1)
    print(f"  Advanced Assembler (Crankshaft)  ×{adv_assemblers:>3}")
    print(f"  (All other machines stay the same)")
    print(f"  TOTAL MACHINES: {opt_total}  (saved {total_machines - opt_total} assemblers)")
    print()

    # ── Bottleneck Analysis ───────────────────────────────────
    print_header("BOTTLENECK ANALYSIS — COAL VERSION")
    stages = [
        ("Iron Drill",       iron_drills * drill_rate,      raw_iron_demand),
        ("Coal Drill",       coal_drills * coal_drill_rate,  coal_demand),
        ("Elec. Furnace",    elec_furnaces * furnace_rate,   liquid_iron_demand),
        ("Ingot Molder",     molders * molder_rate,          iron_ingot_demand),
        ("Blast Furnace",    blast_furnaces * blast_rate,    steel_ingot_demand),
        ("Roller (Rod)",     rollers_rod * rod_rate,         steel_rod_demand),
        ("Press (Plate)",    presses * plate_rate,           plate_demand),
        ("Roller (Gear)",    rollers_gear * gear_rate,       gear_demand),
        ("Assembler",        crank_assemblers * crank_rate,  TARGET_CRANKSHAFTS_PER_SEC),
    ]
    print(f"  {'Stage':<20} {'Supply/s':>10} {'Demand/s':>10} {'Margin':>10}")
    print("  " + "─" * 52)
    for name, supply, demand in stages:
        margin = supply - demand
        status = "✓" if margin >= 0 else "✗ BOTTLENECK"
        print(f"  {name:<20} {supply:>10.3f} {demand:>10.3f} {margin:>+10.3f}  {status}")
    print()

    # ══════════════════════════════════════════════════════════
    #  COKE FUEL VARIANT
    # ══════════════════════════════════════════════════════════
    print_coke_fuel_variant(total_machines)


# ──────────────────────────────────────────────────────────────
# 2b. COKE FUEL VARIANT CALCULATIONS
#     Blast Furnace: 1 Coke Fuel + 2 Iron Ingots → 6 Steel (5s)
#     Coke Fuel:     4 Coal + 2 Oak Logs → 6 Coke Fuel (5s)
# ──────────────────────────────────────────────────────────────

# Steel demand stays the same: 4.0 steel ingots/sec
# But now: 1 Coke + 2 Iron → 6 Steel per 5s cycle = 1.2 steel/s per furnace

cf_blast_rate        = items_per_sec(6, 5)                          # 1.2 steel/s per furnace
cf_blast_furnaces    = machines_needed(steel_ingot_demand, cf_blast_rate)  # ceil(4.0/1.2) = 4

# Iron Ingot demand with Coke: 2 iron ingots per blast cycle (6 steel)
# 4.0 steel/s ÷ 6 steel/cycle = 0.667 cycles/s → 0.667 × 2 = 1.333 iron/s
cf_iron_ingot_demand = (steel_ingot_demand / 6) * 2                 # ≈ 1.333/s
cf_molders           = machines_needed(cf_iron_ingot_demand, molder_rate)  # ceil(1.333/0.1) = 14

cf_liquid_iron_demand = cf_iron_ingot_demand * 1.8                  # ≈ 2.4/s
cf_elec_furnaces      = machines_needed(cf_liquid_iron_demand, furnace_rate)  # ceil(2.4/0.2) = 12

cf_raw_iron_demand    = cf_liquid_iron_demand                       # ≈ 2.4/s
cf_iron_drills        = machines_needed(cf_raw_iron_demand, drill_rate)  # ceil(2.4/0.0667) = 36

# Coke Fuel demand: 1 coke per blast cycle → 0.667 coke/s
cf_coke_demand        = steel_ingot_demand / 6                      # ≈ 0.667 coke/s

# Coke Fuel production: 4 Coal + 2 Oak Logs → 6 Coke Fuel in 5s
cf_coke_rate          = items_per_sec(6, 5)                         # 1.2 coke/s per furnace
cf_coke_furnaces      = machines_needed(cf_coke_demand, cf_coke_rate)  # ceil(0.667/1.2) = 1

# Coal demand for coke production: 4 coal per coke cycle
# 0.667 coke/s ÷ 6 coke/cycle × 4 coal/cycle = 0.444 coal/s
cf_coal_demand        = (cf_coke_demand / 6) * 4                    # ≈ 0.444/s
cf_coal_drills        = machines_needed(cf_coal_demand, coal_drill_rate)  # ceil(0.444/0.0667) = 7

# Oak Log demand: 2 oak logs per coke cycle
# 0.667 coke/s ÷ 6 coke/cycle × 2 logs/cycle = 0.222 logs/s
cf_oak_demand         = (cf_coke_demand / 6) * 2                    # ≈ 0.222/s
# Tree Farm: 1 tree → 2 oak logs, tree grows in ~30-60s depending on sprinkler
# Assuming ~45s per tree with sprinkler, 2 logs/tree = 0.044 logs/s per tree
cf_tree_rate          = items_per_sec(2, 45)                        # ≈ 0.044 logs/s per tree
cf_trees              = machines_needed(cf_oak_demand, cf_tree_rate) # ceil(0.222/0.044) = 6


def print_coke_fuel_variant(coal_total):
    print()
    print_header("🔥 COKE FUEL VARIANT — MASSIVELY MORE EFFICIENT")
    print(f"  Blast Furnace recipe: 1 Coke Fuel + 2 Iron Ingots → 6 Steel")
    print(f"  Coke Fuel recipe:     4 Coal + 2 Oak Logs → 6 Coke Fuel")
    print(f"  Result: 3× more steel per cycle, drastically fewer drills!")
    print()

    cf_total = (cf_iron_drills + cf_coal_drills + cf_elec_furnaces +
                cf_molders + cf_blast_furnaces + cf_coke_furnaces +
                rollers_rod + presses + rollers_gear +
                crank_assemblers + cf_trees + 1)  # +1 for RS2

    print("─" * 62)
    print("  STAGE                               QTY  │ THROUGHPUT")
    print("─" * 62)

    print()
    print("  ▸ MINING")
    print_row("Iron Drill", cf_iron_drills, drill_rate, cf_iron_drills * drill_rate, "(raw iron)")
    print_row("Coal Drill", cf_coal_drills, coal_drill_rate, cf_coal_drills * coal_drill_rate, "(coal → coke)")

    print()
    print("  ▸ TREE FARM (for Oak Logs)")
    print_row("Trees + Harvester", cf_trees, cf_tree_rate, cf_trees * cf_tree_rate, "(oak logs)")

    print()
    print("  ▸ SMELTING & INGOTS")
    print_row("Electric Furnace", cf_elec_furnaces, furnace_rate, cf_elec_furnaces * furnace_rate, "(liquid iron)")
    print_row("Ingot Molder", cf_molders, molder_rate, cf_molders * molder_rate, "(iron ingots)")
    print_row("Blast Furnace (Coke→Steel)", cf_blast_furnaces, cf_blast_rate, cf_blast_furnaces * cf_blast_rate, "(steel ingots)")
    print_row("Blast Furnace (Coke Fuel)", cf_coke_furnaces, cf_coke_rate, cf_coke_furnaces * cf_coke_rate, "(coke fuel)")

    print()
    print("  ▸ STEEL PROCESSING (same as coal version)")
    print_row("Roller (Steel Rod)", rollers_rod, rod_rate, rollers_rod * rod_rate, "(steel rods)")
    print_row("Press (Steel Plate)", presses, plate_rate, presses * plate_rate, "(for gears)")
    print_row("Roller (Gear)", rollers_gear, gear_rate, rollers_gear * gear_rate, "(gears)")

    print()
    print("  ▸ ASSEMBLY & RESEARCH")
    print_row("Craft Assembler (Crankshaft)", crank_assemblers, crank_rate, crank_assemblers * crank_rate, "(crankshafts)")
    print_row("Research Station 2", 1, TARGET_CRANKSHAFTS_PER_SEC, TARGET_CRANKSHAFTS_PER_SEC, "(consuming)")

    print()
    print("─" * 62)
    print(f"  TOTAL MACHINES: {cf_total}")
    print("─" * 62)

    # ── Comparison ────────────────────────────────────────────
    print()
    print_header("⚡ COAL vs COKE FUEL — COMPARISON")
    print(f"  {'':30} {'COAL':>10} {'COKE FUEL':>10} {'SAVED':>10}")
    print("  " + "─" * 62)
    comparisons = [
        ("Iron Drills",        iron_drills,    cf_iron_drills),
        ("Coal Drills",        coal_drills,    cf_coal_drills),
        ("Electric Furnaces",  elec_furnaces,  cf_elec_furnaces),
        ("Ingot Molders",      molders,        cf_molders),
        ("Blast Furnaces",     blast_furnaces, cf_blast_furnaces + cf_coke_furnaces),
        ("Rollers (Rod)",      rollers_rod,    rollers_rod),
        ("Presses (Plate)",    presses,        presses),
        ("Rollers (Gear)",     rollers_gear,   rollers_gear),
        ("Assemblers",         crank_assemblers, crank_assemblers),
        ("Trees/Harvester",    0,              cf_trees),
        ("Research Station 2", 1,              1),
    ]
    coal_t = 0
    coke_t = 0
    for name, c_count, k_count in comparisons:
        saved = c_count - k_count
        arrow = f"↓{abs(saved)}" if saved > 0 else (f"↑{abs(saved)}" if saved < 0 else "—")
        print(f"  {name:<30} {c_count:>10} {k_count:>10} {arrow:>10}")
        coal_t += c_count
        coke_t += k_count

    print("  " + "─" * 62)
    print(f"  {'TOTAL':<30} {coal_t:>10} {coke_t:>10} {'↓' + str(coal_t - coke_t):>10}")
    pct = (1 - coke_t / coal_t) * 100
    print(f"\n  🎯 Coke Fuel saves {coal_t - coke_t} machines ({pct:.0f}% reduction)!")

    # ── Coke Fuel Bottleneck Analysis ─────────────────────────
    print()
    print_header("BOTTLENECK ANALYSIS — COKE FUEL VERSION")
    cf_stages = [
        ("Iron Drill",       cf_iron_drills * drill_rate,       cf_raw_iron_demand),
        ("Coal Drill",       cf_coal_drills * coal_drill_rate,   cf_coal_demand),
        ("Trees (Oak Logs)", cf_trees * cf_tree_rate,            cf_oak_demand),
        ("Elec. Furnace",    cf_elec_furnaces * furnace_rate,    cf_liquid_iron_demand),
        ("Ingot Molder",     cf_molders * molder_rate,           cf_iron_ingot_demand),
        ("BF (Coke Fuel)",   cf_coke_furnaces * cf_coke_rate,    cf_coke_demand),
        ("BF (Steel)",       cf_blast_furnaces * cf_blast_rate,  steel_ingot_demand),
        ("Roller (Rod)",     rollers_rod * rod_rate,             steel_rod_demand),
        ("Press (Plate)",    presses * plate_rate,               plate_demand),
        ("Roller (Gear)",    rollers_gear * gear_rate,           gear_demand),
        ("Assembler",        crank_assemblers * crank_rate,      TARGET_CRANKSHAFTS_PER_SEC),
    ]
    print(f"  {'Stage':<20} {'Supply/s':>10} {'Demand/s':>10} {'Margin':>10}")
    print("  " + "─" * 52)
    for name, supply, demand in cf_stages:
        margin = supply - demand
        status = "✓" if margin >= 0 else "✗ BOTTLENECK"
        print(f"  {name:<20} {supply:>10.3f} {demand:>10.3f} {margin:>+10.3f}  {status}")
    print()


# ──────────────────────────────────────────────────────────────
# 4.  REAL-TIME SIMULATOR (optional — runs in terminal)
#     Simulates production over time to verify no stalls.
# ──────────────────────────────────────────────────────────────

import time

def realtime_sim(duration_sec=60, speed=1.0):
    """
    Simulate the full chain for `duration_sec` game-seconds.
    speed=1.0 means real-time, speed=10 means 10× faster.
    """
    print()
    print_header(f"REAL-TIME SIMULATION ({duration_sec}s at {speed}× speed)")

    dt = 0.1  # tick in game-seconds

    # Buffers (item counts between stages)
    buf = {
        "Raw Iron": 0, "Coal": 0, "Liquid Iron": 0,
        "Iron Ingot": 0, "Steel Ingot": 0,
        "Steel Rod": 0, "Steel Plate": 0, "Gear": 0,
        "Crankshaft": 0,
    }

    # Cumulative production counters
    produced = {k: 0.0 for k in buf}
    produced["RP"] = 0.0

    # Production rates (items/sec with given machine counts)
    rates = {
        "Raw Iron":     iron_drills * drill_rate,
        "Coal":         coal_drills * coal_drill_rate,
        "Liquid Iron":  elec_furnaces * furnace_rate,
        "Iron Ingot":   molders * molder_rate,
        "Steel Ingot":  blast_furnaces * blast_rate,
        "Steel Rod":    rollers_rod * rod_rate,
        "Steel Plate":  presses * plate_rate,
        "Gear":         rollers_gear * gear_rate,
        "Crankshaft":   crank_assemblers * crank_rate,
    }

    # Consumption ratios (inputs needed per output)
    # Each step consumes from buffers, produces into buffers
    t = 0.0
    stall_log = []

    while t < duration_sec:
        # STEP 1: Mining (no input needed)
        buf["Raw Iron"] += rates["Raw Iron"] * dt
        buf["Coal"]     += rates["Coal"] * dt

        # STEP 2: Electric Furnace — Raw Iron → Liquid Iron
        can_smelt = min(buf["Raw Iron"], rates["Liquid Iron"] * dt)
        buf["Raw Iron"]    -= can_smelt
        buf["Liquid Iron"] += can_smelt

        # STEP 3: Ingot Molder — 1.8 Liquid Iron → 1 Iron Ingot
        can_mold = min(buf["Liquid Iron"] / 1.8, rates["Iron Ingot"] * dt)
        buf["Liquid Iron"] -= can_mold * 1.8
        buf["Iron Ingot"]  += can_mold

        # STEP 4: Blast Furnace — 1 Iron Ingot + 4 Coal → 2 Steel Ingot
        can_blast_iron = buf["Iron Ingot"]
        can_blast_coal = buf["Coal"] / 4
        can_blast = min(can_blast_iron, can_blast_coal, rates["Steel Ingot"] * dt / 2) # cycles
        buf["Iron Ingot"]  -= can_blast
        buf["Coal"]        -= can_blast * 4
        buf["Steel Ingot"] += can_blast * 2

        # STEP 5a: Roller — 2 Steel Ingot → 1 Steel Rod
        can_rod = min(buf["Steel Ingot"] / 2, rates["Steel Rod"] * dt)
        steel_for_rods = can_rod * 2
        # STEP 5b: Press — 1 Steel Ingot → 1 Steel Plate
        remaining_steel = buf["Steel Ingot"] - steel_for_rods
        can_plate = min(remaining_steel, rates["Steel Plate"] * dt)

        buf["Steel Ingot"] -= (steel_for_rods + can_plate)
        buf["Steel Rod"]   += can_rod
        buf["Steel Plate"] += can_plate

        # STEP 6: Roller — 1 Steel Plate → 1 Gear
        can_gear = min(buf["Steel Plate"], rates["Gear"] * dt)
        buf["Steel Plate"] -= can_gear
        buf["Gear"]        += can_gear

        # STEP 7: Craft Assembler — 2 Steel Rod + 4 Gear → 1 Crankshaft
        can_crank_rod  = buf["Steel Rod"] / 2
        can_crank_gear = buf["Gear"] / 4
        can_crank = min(can_crank_rod, can_crank_gear, rates["Crankshaft"] * dt)
        buf["Steel Rod"] -= can_crank * 2
        buf["Gear"]      -= can_crank * 4
        buf["Crankshaft"] += can_crank

        # STEP 8: Research Station 2 — consumes 0.5 crankshaft/s
        can_research = min(buf["Crankshaft"], TARGET_CRANKSHAFTS_PER_SEC * dt)
        buf["Crankshaft"] -= can_research
        rp_gained = can_research * 35
        produced["RP"] += rp_gained

        # Track total production
        produced["Crankshaft"] += can_crank

        # Check for stalls
        if can_research < TARGET_CRANKSHAFTS_PER_SEC * dt * 0.9:
            stall_log.append(f"  ⚠ t={t:.1f}s: RS2 starved (only {can_research/dt:.3f}/s vs {TARGET_CRANKSHAFTS_PER_SEC}/s)")

        t += dt

        # Print progress every 10 seconds
        if abs(t % 10) < dt or abs(t % 10 - 10) < dt:
            elapsed = t
            print(f"  t={elapsed:>5.0f}s │ Cranks: {produced['Crankshaft']:>7.1f} │ RP: {produced['RP']:>8.1f} │ Buffer: {buf['Crankshaft']:>5.1f}")

        # Real-time pacing
        if speed < 100:
            time.sleep(dt / speed)

    print()
    print(f"  ── Simulation Complete ──")
    print(f"  Total Crankshafts produced: {produced['Crankshaft']:.1f}")
    print(f"  Total RP earned:            {produced['RP']:.1f}")
    print(f"  Average RP/sec:             {produced['RP']/duration_sec:.2f}")
    if stall_log:
        print(f"\n  ⚠ STALLS DETECTED ({len(stall_log)}):")
        for s in stall_log[:10]:
            print(s)
        if len(stall_log) > 10:
            print(f"    ... and {len(stall_log)-10} more")
    else:
        print(f"  ✓ No stalls detected — production chain is balanced!")


# ──────────────────────────────────────────────────────────────
# 5.  ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

    # Run simulation at 100× speed for quick verification
    print("\n  Running 60-second simulation at 100× speed...")
    realtime_sim(duration_sec=60, speed=100)

    print("\n  Run with speed=1.0 for true real-time pacing:")
    print("    realtime_sim(duration_sec=300, speed=1.0)")
    print()