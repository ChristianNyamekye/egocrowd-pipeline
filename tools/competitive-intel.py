"""
Competitive Intelligence Generator for Robotics Data Platform
Generates a markdown briefing comparing our approach to competitors.
Run: python tools/competitive-intel.py
Output: competitive-landscape.md
"""

import json
from datetime import datetime

COMPETITORS = {
    "Lumos (China)": {
        "approach": "Backpack-mounted UMI (FastUMI Pro), 10K units in 2026",
        "funding": "Unknown (Series A likely)",
        "data_method": "UMI gripper with tracking markers, decoupled from robot hardware",
        "cost_per_demo": "$0.08 (RMB 0.6) per task demo",
        "environments": "Industrial, homes, hotels, restaurants, malls, offices",
        "strengths": ["Massive scale (10K units)", "Very low per-demo cost", "UMI proven at Stanford/Columbia/TRI"],
        "weaknesses": ["China-based (export/IP concerns)", "Custom hardware required", "Manipulation only (no whole-body)"],
        "threat_level": "HIGH",
        "url": "https://kr-asia.com/this-company-has-built-a-backpack-style-system-for-robotics-data-collection"
    },
    "RoboStream (Cogito Tech)": {
        "approach": "Managed service — teleoperation + simulation + human oversight",
        "funding": "Part of Cogito Tech (established AI data company)",
        "data_method": "Professional annotators + teleoperation rigs",
        "cost_per_demo": "Enterprise pricing (likely $5-50 per demo)",
        "environments": "Controlled lab + sim environments",
        "strengths": ["Enterprise trust (Cogito brand)", "Quality control via human oversight", "End-to-end service"],
        "weaknesses": ["Expensive", "Not scalable to millions of demos", "Managed service = slow iteration"],
        "threat_level": "MEDIUM",
        "url": "https://www.hackster.io/robostream/robotics-data-collection-cd157e"
    },
    "NVIDIA EgoScale": {
        "approach": "Egocentric human video + retargeted dexterous hand actions",
        "funding": "NVIDIA internal (unlimited)",
        "data_method": "20,854h egocentric video, flow-based VLA, mid-training alignment",
        "cost_per_demo": "$8,700+ capture rig",
        "environments": "Research/lab",
        "strengths": ["Massive scale (20K+ hours)", "VLA architecture (state of art)", "NVIDIA compute backing"],
        "weaknesses": ["Research paper, not product", "Expensive capture rig", "Not open for external data"],
        "threat_level": "LOW (research, not product)",
        "url": "https://arxiv.org/abs/2602.16710"
    },
    "Open X-Embodiment": {
        "approach": "Community dataset — 1M+ trajectories, 22 robot types",
        "funding": "Google DeepMind + academic consortium",
        "data_method": "Aggregated from 20+ labs, heterogeneous collection methods",
        "cost_per_demo": "Free (open dataset)",
        "environments": "Lab environments",
        "strengths": ["Largest open dataset", "Cross-embodiment", "Industry standard format"],
        "weaknesses": ["Lab-only data", "No real-world diversity", "Static — not continuously growing"],
        "threat_level": "LOW (complementary, not competitive)",
        "url": "https://robotics-transformer-x.github.io/"
    },
    "LeRobot (Hugging Face)": {
        "approach": "Unified robot learning framework + community dataset hosting",
        "funding": "Hugging Face backing",
        "data_method": "Community-contributed, standardized format (Parquet + MP4)",
        "cost_per_demo": "Free platform, data from community",
        "environments": "Varied (community-contributed)",
        "strengths": ["HuggingFace ecosystem", "Standard format", "Growing community", "Open source"],
        "weaknesses": ["No capture hardware", "Quality varies", "Hobbyist-scale data"],
        "threat_level": "MEDIUM (platform play, could eat our distribution)",
        "url": "https://huggingface.co/lerobot"
    }
}

OUR_APPROACH = {
    "name": "EgoDex (working name)",
    "approach": "Crowdsourced human manipulation data via consumer hardware (iPhone + Apple Watch + data glove)",
    "cost_per_kit": "~$950",
    "cost_per_demo": "Est. $0.50-2.00 (crowdsourced labor + kit amortization)",
    "data_method": "iPhone (egocentric video + ARKit depth) + Apple Watch (wrist IMU/6-DoF) + UDCAP glove (21-joint finger tracking)",
    "formats": "OXE-compatible, LeRobot-compatible, RLDS",
    "differentiators": [
        "Consumer hardware = 10x cheaper than lab rigs",
        "Cross-embodiment retargeting built-in (Allegro, LEAP, Shadow, humanoid hands)",
        "Real-world environments (homes, offices, kitchens — not labs)",
        "Crowdsourceable — anyone with the kit can contribute",
        "OXE/LeRobot format from day one",
        "MuJoCo sim validation pipeline (BC policy training, proven to work)"
    ],
    "current_status": "Pipeline built + sim validated. Hardware kit specced. Outreach to robot companies active.",
    "key_risk": "Glove not retail yet (enterprise orders only). Phone-only capture (no glove) is possible with hand pose estimation."
}


def generate_briefing():
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    lines.append(f"# Competitive Landscape — Robotics Data Collection")
    lines.append(f"*Generated: {now}*\n")
    
    # Our position
    lines.append("## Our Position: EgoDex")
    lines.append(f"**Approach:** {OUR_APPROACH['approach']}")
    lines.append(f"**Kit Cost:** {OUR_APPROACH['cost_per_kit']} | **Per-Demo:** {OUR_APPROACH['cost_per_demo']}")
    lines.append(f"**Formats:** {OUR_APPROACH['formats']}")
    lines.append(f"**Status:** {OUR_APPROACH['current_status']}")
    lines.append("\n**Key Differentiators:**")
    for d in OUR_APPROACH['differentiators']:
        lines.append(f"- {d}")
    
    # Competitor table
    lines.append("\n## Competitor Comparison\n")
    lines.append("| Company | Method | Cost/Demo | Threat | Key Advantage |")
    lines.append("|---------|--------|-----------|--------|---------------|")
    for name, c in COMPETITORS.items():
        key_adv = c['strengths'][0] if c['strengths'] else "N/A"
        lines.append(f"| {name} | {c['data_method'][:50]}... | {c['cost_per_demo']} | {c['threat_level']} | {key_adv} |")
    
    # Detailed competitor profiles
    lines.append("\n## Detailed Profiles\n")
    for name, c in COMPETITORS.items():
        lines.append(f"### {name}")
        lines.append(f"**Approach:** {c['approach']}")
        lines.append(f"**Cost:** {c['cost_per_demo']}")
        lines.append(f"**Threat:** {c['threat_level']}")
        lines.append(f"\n**Strengths:** {', '.join(c['strengths'])}")
        lines.append(f"**Weaknesses:** {', '.join(c['weaknesses'])}")
        lines.append(f"**Source:** {c['url']}\n")
    
    # Strategic implications
    lines.append("## Strategic Implications\n")
    lines.append("### Lumos is the biggest threat")
    lines.append("- 10K UMI units = massive data volume advantage by end of 2026")
    lines.append("- BUT: China-based, UMI is manipulation-only, custom hardware")
    lines.append("- **Our counter:** Real-world diversity (homes not labs), cross-embodiment, consumer hardware")
    lines.append("")
    lines.append("### LeRobot is the platform to integrate with, not fight")
    lines.append("- Their format is becoming the standard. We output to it.")
    lines.append("- Partnership opportunity: become a data source for LeRobot ecosystem")
    lines.append("")
    lines.append("### Phone-only capture is the unlock")
    lines.append("- If we can get good data from iPhone alone (no glove), cost drops to ~$0")
    lines.append("- arXiv 2602.09013: RGB videos → 3D hand reconstruction → robot actions is proven")
    lines.append("- This makes us uniquely scalable — billions of phones, zero hardware cost")
    lines.append("")
    lines.append("### Key metric to win: demos per dollar in real-world environments")
    lines.append("- Lab data is solved (OXE). Real-world data at scale is the gap.")
    lines.append("- Our positioning: cheapest real-world manipulation data, any environment, any robot.")
    
    return "\n".join(lines)


if __name__ == "__main__":
    briefing = generate_briefing()
    out_path = "competitive-landscape.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(briefing)
    print(f"✅ Generated {out_path} ({len(briefing)} chars)")
    print("\nKey takeaways:")
    print("  🔴 Lumos (HIGH threat): 10K UMI units, $0.08/demo, China-based")
    print("  🟡 LeRobot (MEDIUM): Platform play, integrate don't fight")
    print("  🟡 RoboStream (MEDIUM): Enterprise managed service, expensive")
    print("  🟢 NVIDIA EgoScale (LOW): Research paper, not product")
    print("  🟢 OXE (LOW): Complementary open dataset")
    print("  💡 Phone-only capture = biggest strategic unlock")
