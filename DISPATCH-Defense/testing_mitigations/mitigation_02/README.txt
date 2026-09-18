TESTING MITIGATIONS / MITIGATION_02
==================================

10-image DIAGNOSTIC: Automatic DISPATCH vs Known-Location LDM.
LDM input = attacked/patched images only. Clean images are evaluation-only.

This is NOT the original 30-image DISPATCH run.
This does NOT create DISPATCH-Defense\mitigations\mitigation_02.

FROZEN / READ-ONLY original experiment:
  D:\project CS\DISPATCH-Defense\mitigations\mitigation_01

Completed previous diagnostic (leave as-is):
  D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_01

This experiment:
  D:\project CS\DISPATCH-Defense\testing_mitigations\mitigation_02

Branches (fully self-contained; no shared source/results):
  automatic\
  known_location\

10 mixed images: 8 attack-success (4 car + 4 person) and 2 controls (1 car + 1 person).
attack_02 only contains person and car; restoration code is not class-hardcoded.
