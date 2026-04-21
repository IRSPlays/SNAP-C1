"""
Generate a HELD-OUT evaluation suite for Nexus-R V1.
These examples NEVER appear in training data.

Categories:
  1. In-distribution rephrased (50) - same facts, new phrasings
  2. OOD topics (50) - facts NOT in training at all
  3. Compositional/multi-step (50) - combining learned facts
  4. Adversarial/edge cases (50) - tricky formatting, negation, near-miss
"""
import json
import os
import random

random.seed(99)  # Different seed from training
eval_examples = []


# ============================================================
# 1. IN-DISTRIBUTION REPHRASED — same facts, novel phrasing
# ============================================================

id_rephrased = [
    # Math — same operations, different wording
    ("If I add two and three, what result do I get?", "5"),
    ("Two plus three equals what?", "5"),
    ("Sum of 7 and 8?", "15"),
    ("How much is seven plus eight?", "15"),
    ("What do you get when you add 6 and 4?", "10"),
    ("9 added to 3 is?", "12"),
    ("What number is 11 minus 4?", "7"),
    ("Take away 5 from 12.", "7"),
    ("If you subtract 3 from 10, what remains?", "7"),
    ("Product of 6 and 7?", "42"),
    ("6 multiplied by 7 gives?", "42"),
    ("What does 8 times 3 equal?", "24"),
    ("How much is 9 times 9?", "81"),
    ("Divide 24 by 6.", "4"),
    ("What is the result of 36 divided by 4?", "9"),

    # Geography — same countries, different question style
    ("In which city can you find the French government?", "Paris."),
    ("Tokyo is found in which country?", "Japan."),
    ("If I travel to Australia, what is the capital city I would visit?", "Canberra."),
    ("Which city serves as Brazil's capital?", "Brasilia."),
    ("Where does the Indian government sit?", "New Delhi."),
    ("The capital city of Germany is called what?", "Berlin."),
    ("Russia's governmental seat is in which city?", "Moscow."),
    ("What city is Canada's capital?", "Ottawa."),

    # Programming — same concepts, different phrasing
    ("In coding, what do we call a named storage location?", "A variable stores a value in a program. Example: x = 5."),
    ("What is the purpose of a function in software?", "A function is a reusable block of code. Example: def add(a, b): return a + b."),
    ("How would a beginner print text in Python?", "print('Hello World')."),
    ("Explain loops to a beginner.", "A loop repeats code. Example: for i in range(5): print(i)."),
    ("What data structure uses key-value pairs?", "A dictionary maps keys to values. Example: d = {'name': 'Alice', 'age': 30}."),

    # Science — same topics, rephrased
    ("How does photosynthesis work?", "Photosynthesis converts sunlight, CO2, and water into glucose and oxygen in plants."),
    ("How fast does light travel?", "The speed of light is about 300,000 km/s."),
    ("What holds us on the ground?", "Gravity pulls objects toward each other. On Earth, it is about 9.8 m/s squared."),
    ("At what temperature does water start to boil?", "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit."),

    # General knowledge — same facts, rephrased
    ("Name the world's tallest peak.", "Mount Everest is the tallest mountain at 8,849 meters."),
    ("What is the biggest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
    ("How many continents exist?", "There are 7 continents: Africa, Antarctica, Asia, Australia, Europe, North America, South America."),
    ("Who is credited with creating the telephone?", "Alexander Graham Bell invented the telephone in 1876."),
    ("Who was the playwright behind Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("In what year did WW2 come to an end?", "World War 2 ended in 1945."),
    ("List all the planets.", "The 8 planets are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune."),
    ("What is a normal human body temperature?", "Normal body temperature is about 37 degrees Celsius or 98.6 Fahrenheit."),

    # More math rephrased
    ("Compute 5 squared.", "25"),
    ("What is twelve squared?", "144"),
    ("3 times 3 times 3?", "27"),
    ("Half of 20?", "10"),
    ("Double 6.", "12"),
    ("Triple 4.", "12"),
    ("One hundred divided by 10?", "10"),
]

for q, a in id_rephrased:
    eval_examples.append({"instruction": q, "output": a, "category": "id_rephrased"})

print(f"ID Rephrased: {len(id_rephrased)}")


# ============================================================
# 2. OOD TOPICS — facts NOT in training data at all
# ============================================================

ood_facts = [
    # Countries not in training geography
    ("What is the capital of Iceland?", "Reykjavik."),
    ("What is the capital of Croatia?", "Zagreb."),
    ("What is the capital of Romania?", "Bucharest."),
    ("What is the capital of Cuba?", "Havana."),
    ("What is the capital of Ethiopia?", "Addis Ababa."),
    ("What is the capital of Morocco?", "Rabat."),
    ("What is the capital of Ukraine?", "Kyiv."),
    ("What is the capital of Philippines?", "Manila."),
    ("What is the capital of Pakistan?", "Islamabad."),
    ("What is the capital of Iraq?", "Baghdad."),

    # Math outside training range
    ("What is 25 + 37?", "62"),
    ("What is 99 - 47?", "52"),
    ("What is 15 times 13?", "195"),
    ("What is 144 divided by 12?", "12"),
    ("What is 20 squared?", "400"),
    ("What is 50 + 50?", "100"),
    ("What is 100 - 1?", "99"),
    ("What is 7 times 15?", "105"),
    ("What is 81 divided by 9?", "9"),
    ("What is 13 squared?", "169"),

    # Science not in training
    ("What is the chemical formula for water?", "H2O."),
    ("What planet is closest to the Sun?", "Mercury."),
    ("What is the largest planet in our solar system?", "Jupiter."),
    ("How many teeth does an adult human have?", "32 teeth."),
    ("What is the hardest natural substance?", "Diamond."),
    ("What gas do humans breathe in?", "Oxygen."),
    ("What is the chemical symbol for gold?", "Au."),
    ("How many chromosomes do humans have?", "46 chromosomes (23 pairs)."),

    # Programming not in training
    ("What does HTML stand for?", "HyperText Markup Language."),
    ("What is a compiler?", "A compiler translates source code into machine code."),
    ("What does CPU stand for?", "Central Processing Unit."),
    ("What is an operating system?", "Software that manages computer hardware and runs programs."),
    ("What is RAM?", "Random Access Memory - temporary storage for running programs."),
    ("What is a database?", "An organized collection of structured data stored electronically."),
    ("What language runs in web browsers?", "JavaScript."),
    ("What does URL stand for?", "Uniform Resource Locator."),

    # General knowledge not in training
    ("What is the largest desert in the world?", "The Sahara Desert."),
    ("What is the smallest country in the world?", "Vatican City."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
    ("How many days are in a year?", "365 days (366 in a leap year)."),
    ("What is the fastest land animal?", "The cheetah."),
    ("What year did humans first land on the moon?", "1969."),
    ("What is the tallest animal?", "The giraffe."),
    ("How many hours are in a day?", "24 hours."),
    ("What is the largest organ in the human body?", "The skin."),
    ("What is the speed of sound?", "About 343 meters per second in air."),
]

for q, a in ood_facts:
    eval_examples.append({"instruction": q, "output": a, "category": "ood"})

print(f"OOD: {len(ood_facts)}")


# ============================================================
# 3. COMPOSITIONAL / MULTI-STEP
# ============================================================

compositional = [
    # Multi-step math
    ("What is 2 + 3 + 4?", "9"),
    ("What is 5 times 2 plus 3?", "13"),
    ("What is 10 minus 3 minus 2?", "5"),
    ("What is 4 times 4 minus 6?", "10"),
    ("What is 3 squared plus 1?", "10"),
    ("What is 2 times 3 times 4?", "24"),
    ("What is 100 divided by 5 divided by 2?", "10"),
    ("What is 8 plus 8 plus 8?", "24"),
    ("If I have 12 apples and give away 5, then get 3 more, how many do I have?", "10"),
    ("What is 6 times 6 divided by 4?", "9"),

    # Word math problems
    ("If a book costs 5 dollars and you buy 3, how much total?", "15 dollars."),
    ("I have 20 cookies and eat 7. How many are left?", "13 cookies."),
    ("A class has 12 boys and 8 girls. How many students total?", "20 students."),
    ("Each box has 6 eggs. I have 4 boxes. How many eggs total?", "24 eggs."),
    ("I scored 8 out of 10 on a test. What percentage is that?", "80 percent."),

    # Cross-domain
    ("Is Paris in Europe?", "Yes, Paris is in France, which is in western Europe."),
    ("How many capitals do you know in Europe?", "Many, including Paris, Berlin, Rome, Madrid, London, and more."),
    ("Name a capital city in South America.", "Brasilia (Brazil), Buenos Aires (Argentina), or Lima (Peru)."),
    ("What is the capital of the largest country in the world?", "Moscow is the capital of Russia, the largest country."),
    ("Is Tokyo in Europe or Asia?", "Asia. Tokyo is in Japan, which is in East Asia."),

    # Programming chains
    ("If x = 5 and y = 3, what is x + y?", "8"),
    ("If x = 10 and y = x / 2, what is y?", "5"),
    ("Write code to print the numbers 1 to 5.", "for i in range(1, 6): print(i)"),
    ("What is the output of print(2 * 3)?", "6"),
    ("What is the output of print('Hello' + ' ' + 'World')?", "Hello World"),

    # Science reasoning
    ("Is the speed of light faster than the speed of sound?", "Yes, light is about 300,000 km/s while sound is about 343 m/s."),
    ("Which is heavier, a kilogram of steel or a kilogram of feathers?", "They weigh the same - both are one kilogram."),
    ("If water boils at 100C, does it boil at 90C?", "No, water does not boil at 90 degrees Celsius."),

    # Multi-fact
    ("Name two countries in East Asia and their capitals.", "Japan (Tokyo) and China (Beijing)."),
    ("What are two things photosynthesis needs?", "Sunlight and CO2 (carbon dioxide)."),
    ("Name a planet smaller than Earth.", "Mercury or Mars."),

    # Math with context
    ("What is 7 times 8? Give the answer as a number.", "56"),
    ("Calculate: 11 + 12 = ?", "23"),
    ("If 5 squared is 25, what is 6 squared?", "36"),
    ("What comes after 12 squared (144)?", "13 squared is 169."),
    ("Is 7 times 7 equal to 48?", "No, 7 times 7 is 49."),

    # Ordering / comparison
    ("Which is bigger, 7 times 8 or 9 times 6?", "7 times 8 is 56 and 9 times 6 is 54, so 7 times 8."),
    ("Is the Pacific or Atlantic Ocean larger?", "The Pacific Ocean is larger."),
    ("Is Python a compiled or interpreted language?", "Python is an interpreted language."),
    ("Which is colder: 0C or 32F?", "They are the same temperature."),

    # Misc compositional
    ("What letter does the capital of France start with?", "P (Paris)."),
    ("How many letters are in the word 'hello'?", "5 letters."),
    ("Spell the number 42.", "Forty-two."),
    ("What is half of a dozen?", "6."),
    ("How many seconds in a minute?", "60 seconds."),
    ("How many minutes in an hour?", "60 minutes."),
    ("How many months have 30 days?", "4 months: April, June, September, November."),
    ("What day comes after Monday?", "Tuesday."),
    ("Is 15 an odd or even number?", "Odd."),
    ("What is the next prime number after 7?", "11."),
]

for q, a in compositional:
    eval_examples.append({"instruction": q, "output": a, "category": "compositional"})

print(f"Compositional: {len(compositional)}")


# ============================================================
# 4. ADVERSARIAL / EDGE CASES
# ============================================================

adversarial = [
    # Near miss / wrong answers
    ("Is Berlin the capital of France?", "No, Paris is the capital of France. Berlin is the capital of Germany."),
    ("Is 2 + 2 = 5?", "No, 2 + 2 = 4."),
    ("Is Tokyo the capital of China?", "No, Beijing is the capital of China. Tokyo is the capital of Japan."),
    ("What is 0 times 999?", "0"),
    ("What is 1 times 1?", "1"),
    ("What is 0 + 0?", "0"),
    ("What is 12 - 12?", "0"),

    # Unusual formatting
    ("WHAT IS 5+5?", "10"),
    ("what is the capital of japan?", "Tokyo."),
    ("capital of france", "Paris."),
    ("2+3", "5"),
    ("10*5", "50"),
    ("hello", "Hello! How can I help you today?"),

    # Trick questions
    ("What is the capital of the Moon?", "The Moon does not have a capital city."),
    ("What is 5 divided by 0?", "Division by zero is undefined."),
    ("Who is the king of the United States?", "The United States does not have a king. It has a president."),
    ("What color is the number 7?", "Numbers do not have colors."),

    # Boundary math
    ("What is 1 + 0?", "1"),
    ("What is 0 squared?", "0"),
    ("What is 1 squared?", "1"),
    ("What is 12 times 12?", "144"),
    ("What is 12 + 12?", "24"),
    ("What is 1 - 1?", "0"),

    # Ambiguous
    ("What is Python?", "Python is a programming language."),
    ("What is a bug?", "In programming, a bug is an error in code."),
    ("What is Java?", "Java is a programming language."),

    # Repeated / redundant
    ("What is 2 + 3? What is the answer?", "5"),
    ("Tell me, please, what is the capital of France?", "Paris."),
    ("I need to know what 10 times 5 is.", "50"),

    # Very short
    ("Hi", "Hello! How can I help you today?"),
    ("Thanks", "You're welcome! Let me know if you need anything else."),
    ("?", "I'm sorry, could you please ask a question?"),

    # Empty-ish
    ("What is a?", "Could you clarify what you mean by 'a'?"),
    ("Why?", "Could you provide more context?"),

    # Negation
    ("What is NOT the capital of France?", "Any city other than Paris is not the capital of France."),
    ("Name something that is not a planet.", "The Moon, the Sun, or any star."),
    ("What is not a programming language?", "English is a natural language, not a programming language."),

    # Multi-part
    ("What is 3 + 3 and what is 4 + 4?", "3 + 3 = 6 and 4 + 4 = 8."),
    ("Name a European capital and an Asian capital.", "Paris (France) and Tokyo (Japan)."),

    # Numbers as words
    ("What is three plus four?", "7"),
    ("What is five times six?", "30"),
    ("What is ten minus two?", "8"),
    ("What is nine divided by three?", "3"),

    # Weird spacing/punctuation
    ("What  is  2 + 3 ?", "5"),
    ("Capital of France??", "Paris."),
    ("What is 10 times 5!!!", "50"),
    ("...what is a function?", "A function is a reusable block of code."),
]

for q, a in adversarial:
    eval_examples.append({"instruction": q, "output": a, "category": "adversarial"})

print(f"Adversarial: {len(adversarial)}")


# ============================================================
# SAVE
# ============================================================

random.shuffle(eval_examples)

output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "eval_suite.jsonl")

with open(output_path, "w", encoding="utf-8") as f:
    for item in eval_examples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Stats
from collections import Counter
cats = Counter(item["category"] for item in eval_examples)
print(f"\nTotal eval examples: {len(eval_examples)}")
for cat, count in sorted(cats.items()):
    print(f"  {cat}: {count}")
print(f"\nSaved to: {output_path}")
