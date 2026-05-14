"""
data_init.py — Run once on startup.
Generates the 5 pharmaceutical drug label images used by the multimodal pipeline
(replicates Notebook 3 Step 4 exactly) and seeds any missing data directories.
"""

import os
from pathlib import Path


def generate_drug_labels():
    """Generate 5 realistic drug label JPGs using PIL — mirrors Notebook 3 exactly."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[data_init] Pillow not installed — skipping drug label generation")
        return

    out_dir = Path("./data/drug_labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already generated
    if (out_dir / "metformin_500mg.jpg").exists():
        print("[data_init] Drug label images already exist — skipping")
        return

    def _draw_label(filename, drug_name, generic_name, dosage, form,
                    manufacturer, lot, expiry, storage,
                    indications, dosage_instructions, warnings,
                    active_ingredient, inactive_ingredients,
                    header_color=(0, 70, 127)):
        W, H = 820, 1060
        img = Image.new("RGB", (W, H), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)

        # Header band
        draw.rectangle([0, 0, W, 110], fill=header_color)
        draw.text((20, 15), drug_name, fill=(255, 255, 255))
        draw.text((20, 55), generic_name, fill=(220, 220, 220))
        draw.text((20, 80), f"{dosage}  |  {form}", fill=(200, 230, 255))

        # RX only badge
        draw.rectangle([W - 130, 15, W - 15, 55], fill=(200, 0, 0))
        draw.text((W - 120, 25), "Rx ONLY", fill=(255, 255, 255))

        # Body
        y = 130
        sections = [
            ("MANUFACTURER", manufacturer),
            ("LOT / EXPIRY", f"Lot: {lot}   Exp: {expiry}"),
            ("STORAGE", storage),
            ("INDICATIONS", indications),
            ("DOSAGE INSTRUCTIONS", dosage_instructions),
            ("WARNINGS", warnings),
            ("ACTIVE INGREDIENT", active_ingredient),
            ("INACTIVE INGREDIENTS", inactive_ingredients),
        ]
        for title, body in sections:
            draw.rectangle([15, y, W - 15, y + 22], fill=header_color)
            draw.text((20, y + 4), title, fill=(255, 255, 255))
            y += 28
            # Word-wrap body text
            words = body.split()
            line = ""
            for word in words:
                test = line + " " + word if line else word
                if len(test) > 90:
                    draw.text((20, y), line, fill=(40, 40, 40))
                    y += 18
                    line = word
                else:
                    line = test
            if line:
                draw.text((20, y), line, fill=(40, 40, 40))
            y += 24

        # Footer
        draw.rectangle([0, H - 40, W, H], fill=(220, 220, 220))
        draw.text((20, H - 28), f"Keep out of reach of children. Store as directed.", fill=(80, 80, 80))

        path = out_dir / filename
        img.save(str(path), "JPEG", quality=90)
        print(f"[data_init] Generated {path}")

    drugs = [
        dict(
            filename="metformin_500mg.jpg",
            drug_name="GLUCOPHAGE",
            generic_name="Metformin Hydrochloride Tablets",
            dosage="500 mg",
            form="Film-coated Tablets",
            manufacturer="Bristol-Myers Squibb",
            lot="BMS20241105",
            expiry="2026-08",
            storage="Store below 25°C. Keep dry. Protect from light.",
            indications="Type 2 Diabetes Mellitus. Adjunct to diet and exercise to improve glycemic control.",
            dosage_instructions="Adults: 500 mg twice daily with meals. Max 2550 mg/day. Swallow whole.",
            warnings="Contraindicated in eGFR <30. Risk of lactic acidosis. Discontinue before contrast procedures. Monitor renal function annually.",
            active_ingredient="Metformin Hydrochloride 500 mg",
            inactive_ingredients="Microcrystalline cellulose, povidone, magnesium stearate, hypromellose, polyethylene glycol.",
            header_color=(0, 70, 127),
        ),
        dict(
            filename="amoxicillin_500mg.jpg",
            drug_name="AMOXIL",
            generic_name="Amoxicillin Capsules",
            dosage="500 mg",
            form="Hard Gelatin Capsules",
            manufacturer="GlaxoSmithKline",
            lot="GSK20241210",
            expiry="2026-03",
            storage="Store below 25°C in dry conditions. Keep away from moisture.",
            indications="Bacterial infections: respiratory tract, urinary tract, skin, ear, nose and throat infections caused by susceptible organisms.",
            dosage_instructions="Adults: 250-500 mg every 8 hours. Severe infections: 500 mg every 8 hours for 7-14 days.",
            warnings="Penicillin allergy: do not use. Anaphylaxis risk. Check allergy history before prescribing. C. difficile risk with prolonged use.",
            active_ingredient="Amoxicillin trihydrate equivalent to Amoxicillin 500 mg",
            inactive_ingredients="Magnesium stearate, colloidal silicon dioxide. Capsule shell: gelatin, titanium dioxide.",
            header_color=(0, 110, 60),
        ),
        dict(
            filename="ibuprofen_400mg.jpg",
            drug_name="BRUFEN",
            generic_name="Ibuprofen Tablets",
            dosage="400 mg",
            form="Film-coated Tablets",
            manufacturer="Abbott Laboratories",
            lot="ABT20241101",
            expiry="2026-06",
            storage="Store below 30°C. Keep in original packaging. Protect from moisture.",
            indications="Mild to moderate pain, fever, dysmenorrhoea, musculoskeletal disorders, dental pain, post-operative pain.",
            dosage_instructions="Adults and children over 12: 400 mg 3 times daily with food. Max 1200 mg/day OTC. Do not exceed 2400 mg/day.",
            warnings="NSAID risk: GI bleeding, ulceration, cardiovascular events. Avoid in pregnancy third trimester. Caution in renal/hepatic impairment, elderly, asthmatics.",
            active_ingredient="Ibuprofen 400 mg",
            inactive_ingredients="Lactose monohydrate, microcrystalline cellulose, croscarmellose sodium, magnesium stearate, hypromellose, macrogol, titanium dioxide.",
            header_color=(180, 60, 0),
        ),
        dict(
            filename="atorvastatin_20mg.jpg",
            drug_name="LIPITOR",
            generic_name="Atorvastatin Calcium Tablets",
            dosage="20 mg",
            form="Film-coated Tablets",
            manufacturer="Pfizer Inc.",
            lot="PFZ20241208",
            expiry="2026-11",
            storage="Store below 25°C. Protect from light and moisture.",
            indications="Primary hypercholesterolaemia, mixed dyslipidaemia, prevention of cardiovascular events in high-risk patients.",
            dosage_instructions="Adults: 10-80 mg once daily at any time of day with or without food. Starting dose 10-20 mg. Max 80 mg/day.",
            warnings="Myopathy/rhabdomyolysis risk: report unexplained muscle pain. Hepatotoxicity: monitor LFTs. Contraindicated in active liver disease and pregnancy/breastfeeding.",
            active_ingredient="Atorvastatin calcium equivalent to Atorvastatin 20 mg",
            inactive_ingredients="Calcium carbonate, microcrystalline cellulose, lactose monohydrate, croscarmellose sodium, polysorbate 80, hydroxypropylcellulose, magnesium stearate.",
            header_color=(100, 0, 120),
        ),
        dict(
            filename="paracetamol_500mg.jpg",
            drug_name="PANADOL",
            generic_name="Paracetamol Tablets",
            dosage="500 mg",
            form="Tablets",
            manufacturer="Haleon (formerly GSK Consumer Healthcare)",
            lot="HLN20241115",
            expiry="2027-01",
            storage="Store below 25°C. Keep in original blister. Keep dry.",
            indications="Mild to moderate pain: headache, toothache, backache, period pain, cold and flu symptoms. Fever reduction.",
            dosage_instructions="Adults and children over 12: 500 mg-1000 mg every 4-6 hours. Max 4000 mg (8 tablets) in 24 hours. Do not exceed stated dose.",
            warnings="Do not take with other paracetamol-containing products. Risk of severe liver damage with overdose or alcohol use. Seek medical attention for overdose immediately.",
            active_ingredient="Paracetamol 500 mg",
            inactive_ingredients="Pregelatinised maize starch, potassium sorbate, purified talc, stearic acid.",
            header_color=(0, 100, 160),
        ),
    ]

    for d in drugs:
        try:
            _draw_label(**d)
        except Exception as e:
            print(f"[data_init] Could not generate {d['filename']}: {e}")

    print("[data_init] Drug label generation complete")


def ensure_data_dirs():
    """Make sure all data directories exist."""
    dirs = [
        "./data/health",
        "./data/disease_outbreaks",
        "./data/hospital_reports",
        "./data/clinical_guidelines",
        "./data/drug_labels",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_data_dirs()
    generate_drug_labels()
    print("[data_init] All done.")
