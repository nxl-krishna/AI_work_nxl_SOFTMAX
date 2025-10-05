import random

print("Starting the Authentic Melody Composer for Raag Bhairav...")

# --- Define Authentic Musical Phrases (Taans) ---
phrases = {
    "aroha_start": ['S', 'r', 'G', 'm'],
    "aroha_mid": ['m', 'P', 'd', 'N'],
    "pakar": ['G', 'm', 'd', 'P'],
    "dha_focus": ['d', 'P', 'm', 'P', 'G', 'm'],
    "re_focus": ['G', 'm', 'r', 'S'],
    "avaroha_high": ["S'", 'N', 'd', 'P'],
    "avaroha_low": ['P', 'm', 'G', 'r', 'S'],
    "ending": ['m', 'G', 'r', 'S']
}

# Maps our code notes to their proper, stylized names
note_full_names = {
    'S': 'Sa', 'r': 'Re', 'G': 'Ga', 'm': 'Ma',
    'P': 'Pa', 'd': 'Dha', 'N': 'Ni', "S'": 'Saa'
}

# --- Define an Authentic Rhythmic Cycle ---
rhythm_cycle = [1, 0.5, 0.5, 1,   1, 1, 1, 1,   0.5, 0.5, 0.5, 0.5, 2,   1, 1, 1] 

# --- Set Melody Length ---
target_melody_length = 64
NUMBER_OF_RUNS = 5

# --- Helper Functions ---
def generate_composed_melody(phrase_book, length):
    """
    Builds a melody by intelligently selecting and connecting valid musical phrases.
    """
    melody = list(phrase_book["aroha_start"])
    
    while len(melody) < length:
        last_note = melody[-1]
        possible_phrases = []
        for key, phrase in phrase_book.items():
            if phrase[0] == last_note:
                possible_phrases.append(phrase)
        
        if not possible_phrases:
            next_phrase = random.choice(list(phrase_book.values()))
        else:
            next_phrase = random.choice(possible_phrases)

        melody.extend(next_phrase)

    melody = melody[:length - 4]
    melody.extend(phrase_book["ending"])

    return melody[:length]

def print_rhythmic_melody(melody, rhythm):
    """
    Prints the melody with authentic names and correct rhythmic spelling.
    """
    output = []
    for i, note in enumerate(melody):
        duration = rhythm[i % len(rhythm)]
        base_name = note_full_names[note]
        
        if base_name == 'Ni':
            output.append('Ni')
            continue

        vowel = ''
        for char in reversed(base_name):
            if char.lower() in 'aeiou':
                vowel = char.lower()
                break
        
        if duration == 0.5:
            output.append(base_name)
        elif duration == 1.0:
            output.append(base_name + vowel)
        else:
            output.append(base_name + vowel + vowel)
    
    beats_per_line = 16
    current_beat = 0
    line = []
    for i, item in enumerate(output):
        line.append(item)
        duration = rhythm[i % len(rhythm)]
        current_beat += duration
        
        if current_beat >= beats_per_line:
            print(' '.join(line))
            line = []
            current_beat = 0
    if line:
        print(' '.join(line))

# --- Main Program ---
print(f"Composing an authentic melody of ~{target_melody_length} notes over {NUMBER_OF_RUNS} runs...")

best_melody_so_far = []
# We don't need to score anymore, but we can run multiple times to get variety
for run in range(NUMBER_OF_RUNS):
    print(f"\n--- Composition Run {run + 1}/{NUMBER_OF_RUNS} ---")
    generated_melody = generate_composed_melody(phrases, target_melody_length)
    print_rhythmic_melody(generated_melody, rhythm_cycle)
    # For simplicity, we'll just keep the last one, but you could add a scoring
    # function back in if you wanted to find the "best" of the 5.
    best_melody_so_far = generated_melody

print("\n--- Composition Finished ---")
print("\nFinal selected melody:")
print_rhythmic_melody(best_melody_so_far, rhythm_cycle)