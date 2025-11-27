# model_selection/quick_generate.py

#	•	Imports the HFTextGenerator from core/ and quickly generates one SAT question.
# Purpose: sanity-check inference works locally.


import sys, os
# Add the project root to sys.path to allow imports from 'core' when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.inference import HFTextGenerator

prompt = "Generate 1 SAT-style algebra question with step-by-step answer."
gen = HFTextGenerator()
print(gen.generate(prompt))