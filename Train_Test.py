#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:21:23 2024

@author: marcwhiting
"""

import pandas as pd
import openai
import json
from sklearn.metrics import classification_report

# Set your OpenAI API key
openai.api_key = 'API key'

# Load your Excel file
df = pd.read_excel('/Users/marcwhiting/Desktop/Train 1.xlsx')

# Function to create a single label column based on 'x' marks
def get_label(row):
    if row['Door open'] == 'x':
        return 'Door open'
    elif row['Door Closed'] == 'x':
        return 'Door Closed'
    else:
        return 'Unlabeled'  # For rows that are not labeled

# Apply the function to create the 'Label' column
df['Label'] = df.apply(get_label, axis=1)

# Filter out unlabeled data
df = df[df['Label'] != 'Unlabeled']

# Prepare detailed definitions
definitions = """
Definitions:
- Door open: The learner discusses climate-related impacts that are local to a specific place (e.g., their home community or the museum's region). They have an overall accepting frame, are not critical of others, and are highly engaged in the conversation process.
- Door Closed: The learner expresses skepticism, doom, or criticism regarding climate-related topics. They might be disengaged, critical of others, or display a closed-off attitude towards the conversation.
"""

# Prepare a pool of at least 16 few-shot examples in JSON format
few_shot_examples_pool = [
    """Participant: "I've noticed that the summers here are getting hotter each year, and it's affecting our local crops."
{
  "analysis": "The participant discusses local climate impacts and is engaged in the conversation.",
  "classification": "Door open"
}""",
    """Participant: "Climate change is just a hoax, nothing is really happening."
{
  "analysis": "The participant expresses skepticism about climate change.",
  "classification": "Door Closed"
}""",
    # ... Add more examples to reach at least 16
    # Ensure examples are diverse and representative
]

# Function to create the prompt
def create_prompt(text, num_examples):
    prompt = definitions + "\n\n"
    if num_examples > 0:
        selected_examples = few_shot_examples_pool[:num_examples]
        prompt += "\n\n".join(selected_examples)
        prompt += "\n\n"
    prompt += f"Participant: \"{text}\"\nProvide your analysis and classification in JSON format."
    return prompt

# Function to classify dialogue using OpenAI API
def classify_dialogue(text, num_examples):
    prompt = create_prompt(text, num_examples)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are an assistant that analyzes participant statements based on the definitions and examples provided. Provide your analysis and classification in JSON format with keys 'analysis' and 'classification'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0
        )
        result = response.choices[0].message['content'].strip()
        return result
    except Exception as e:
        print(f"Error classifying text: {e}")
        return None

# Function to extract data from JSON
def extract_from_json(json_str, key):
    try:
        data = json.loads(json_str)
        return data.get(key, None)
    except json.JSONDecodeError:
        return None

# Numbers of examples to test
num_examples_list = [0, 2, 4, 8, 16]

# Dictionary to store results
results = {}

for num_examples in num_examples_list:
    print(f"\nTesting with {num_examples} examples:")
    
    # Apply the classification function to each transcript
    df['PredictedJSON'] = df['Transcript'].apply(lambda x: classify_dialogue(x, num_examples))
    
    # Extract classification and analysis from JSON output
    df['PredictedLabel'] = df['PredictedJSON'].apply(lambda x: extract_from_json(x, 'classification'))
    df['Analysis'] = df['PredictedJSON'].apply(lambda x: extract_from_json(x, 'analysis'))
    
    # Print the DataFrame with actual and predicted labels for debugging
    print(df[['Transcript', 'Label', 'PredictedLabel', 'Analysis']].head())
    
    # Evaluate the model's performance
    print("\nClassification Report:")
    print(classification_report(df['Label'], df['PredictedLabel'], labels=['Door open', 'Door Closed']))
    
    # Store the report for later analysis
    report = classification_report(df['Label'], df['PredictedLabel'], labels=['Door open', 'Door Closed'], output_dict=True)
    results[num_examples] = report
