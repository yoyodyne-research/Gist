# Kitsune Training Data Recording Guide

Thanks for helping collect training data for Kitsune! This guide explains what to record and how.

## Overview

We're training a neural network to recognize playing techniques from audio. Your recordings will help the system learn the difference between fingerpicking, pick attacks, slides, hammer-ons, and other gestures.

**What you need:**
- Your instrument (guitar or bass)
- A way to record clean DI signal (interface, amp modeler, etc.)
- 30-45 minutes total recording time

## Recording Setup

- **Format:** WAV or FLAC, 44.1kHz or 48kHz, mono or stereo
- **Signal:** Clean DI preferred (no effects, no amp sim) — but light compression is fine
- **Level:** Healthy signal, no clipping. Peaks around -6dB is ideal.
- **Environment:** Doesn't need to be silent, but avoid excessive background noise

## Session List

Record each session as a **separate file**. Each session focuses on one technique — just play naturally using that technique for 2-3 minutes. Don't overthink it; variation is good.

### Guitar Sessions

| # | Filename | Technique | Notes |
|---|----------|-----------|-------|
| 1 | `guitar_fingerpick.wav` | Fingerpicking | Arpeggios, Travis picking, classical style — no pick |
| 2 | `guitar_pick_single.wav` | Pick - single notes | Melodies, runs, staccato phrases |
| 3 | `guitar_pick_strum.wav` | Pick - strumming | Chords, varied dynamics (soft to hard) |
| 4 | `guitar_hammer_pull.wav` | Hammer-ons & pull-offs | Legato lines, trills |
| 5 | `guitar_slides.wav` | Slides | Slide into notes, slide between notes |
| 6 | `guitar_palm_mute.wav` | Palm muting | Muted chugs, partial muting |
| 7 | `guitar_bends.wav` | Bends & vibrato | Half-step, whole-step, wide vibrato |
| 8 | `guitar_harmonics.wav` | Harmonics | Natural and artificial/pinch harmonics |
| 9 | `guitar_freeplay.wav` | Free play | Mix everything — play naturally, switch techniques |

### Bass Sessions

| # | Filename | Technique | Notes |
|---|----------|-----------|-------|
| 1 | `bass_finger.wav` | Fingerstyle | Two-finger alternating, walking lines |
| 2 | `bass_slap.wav` | Slap (thumb) | Thumb slaps, muted slaps |
| 3 | `bass_pop.wav` | Pop (finger pull) | Popping strings, slap+pop combos |
| 4 | `bass_pick.wav` | Pick playing | If you play with a pick |
| 5 | `bass_ghost.wav` | Ghost/muted notes | Percussive muted notes, groove patterns |
| 6 | `bass_slides.wav` | Slides | Slide transitions, glissando |
| 7 | `bass_freeplay.wav` | Free play | Mix everything — play naturally |

## Tips

- **Be yourself.** Play in your normal style. We want real playing, not exercises.
- **Vary dynamics.** Soft and loud within each session.
- **Vary tempo.** Fast and slow passages.
- **Vary position.** Different frets, different strings.
- **Mistakes are fine.** Don't stop and restart — just keep going.
- **Free play is important.** This captures natural technique transitions.

## Optional: Extended Sessions

If you have time, these additional sessions would be valuable:

- **Different guitars/basses** — repeat sessions on a different instrument
- **Different pickup positions** — neck vs bridge
- **Specific genres** — blues licks, metal riffs, funk grooves, jazz lines
- **Edge cases** — tapping, whammy bar, behind-the-nut bends, etc.

## File Naming

Please include your name/handle and instrument type:

```
alice_guitar_fingerpick.wav
alice_guitar_pick_single.wav
bob_bass_finger.wav
bob_bass_slap.wav
```

## Delivery

Send files to: [TBD - add your preferred delivery method]

Zip them up or use a file sharing service (Dropbox, Google Drive, WeTransfer, etc.)

## Questions?

Reach out if anything is unclear. Thanks again for contributing!

---

*Recording guide for Kitsune latent layer training data collection*
