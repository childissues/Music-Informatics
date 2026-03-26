"""Team_JS_Bach_MusicGeneration.py

Music Generation — Team J.S. Bach

This script generates our final piano-only MIDI piece by rendering a compact
symbolic specification stored in `data/piece_spec.json`.

Run:
    python Team_JS_Bach_MusicGeneration.py --out_dir output

Output:
    output/Team_JS_Bach_generated.mid
"""

import argparse
import json
import os
from mido import MidiFile, MidiTrack, Message, MetaMessage

def load_spec(spec_path: str) -> dict:
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)

def render(spec: dict, out_path: str, transpose: int = 0, velocity_scale: float = 1.0, tempo_scale: float = 1.0) -> None:
    ticks_per_beat = int(spec["ticks_per_beat"])
    base_tempo = int(spec.get("tempo", 500000))
    tempo = int(base_tempo / tempo_scale) if tempo_scale and tempo_scale != 0 else base_tempo
    program = int(spec.get("program", 0))
    events = spec["events"]

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(Message("program_change", program=program, time=0))

    msgs = []
    for ev in events:
        ch = int(ev.get("ch", 0))
        note = int(ev["note"]) + int(transpose)
        if not (0 <= note <= 127):
            continue
        vel = max(1, min(127, int(round(int(ev.get("vel", 64)) * velocity_scale))))
        start = int(ev["start"])
        end = int(ev["end"])
        msgs.append((start, Message("note_on", note=note, velocity=vel, channel=ch, time=0)))
        msgs.append((end, Message("note_off", note=note, velocity=0, channel=ch, time=0)))

    msgs.sort(key=lambda x: x[0])

    last_t = 0
    for t, m in msgs:
        m.time = max(0, t - last_t)
        last_t = t
        track.append(m)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mid.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--filename", default="Team_JS_Bach_generated.mid")
    ap.add_argument("--transpose", type=int, default=0)
    ap.add_argument("--velocity_scale", type=float, default=1.0)
    ap.add_argument("--tempo_scale", type=float, default=1.0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    spec_path = os.path.join(here, "data", "piece_spec.json")
    spec = load_spec(spec_path)

    out_path = os.path.join(here, args.out_dir, args.filename)
    render(spec, out_path, transpose=args.transpose, velocity_scale=args.velocity_scale, tempo_scale=args.tempo_scale)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
