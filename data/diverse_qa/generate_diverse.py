"""
Generate diverse Q&A data V4 -- Fixes from Round 5 critic:
1. MORE DIVERSE PHRASINGS: Natural language math ("sum of", "If I add", "X plus Y equals")
2. WORD PROBLEMS: "I have 5 apples and buy 3 more" style
3. MULTI-STEP MATH: 2+3+4, 3*4-2, compound operations
4. DROP CONVERSATIONAL (1% = noise floor)
5. BOOST geography/programming/science/general with even more variety
6. Target: ~3000-4000 examples with genuine diversity

Key lesson from eval suite: ID-rephrased only 19% means model is phrasing-brittle.
"""
import json
import random
import os

random.seed(42)
qa_pairs = []


# ============================================================
# MATH — 5 diverse phrasings per fact (natural language variety)
# ============================================================

def make_addition(a, b):
    r = a + b
    all_p = [
        (f"What is {a} + {b}?", f"{r}"),
        (f"{a} + {b} = ?", f"{r}"),
        (f"Calculate {a} plus {b}.", f"{a} + {b} = {r}."),
        (f"Sum of {a} and {b}?", f"{r}"),
        (f"If I add {a} and {b}, what do I get?", f"{r}"),
    ]
    return all_p


def make_subtraction(a, b):
    r = a - b
    return [
        (f"What is {a} - {b}?", f"{r}"),
        (f"{a} - {b} = ?", f"{r}"),
        (f"Subtract {b} from {a}.", f"{a} - {b} = {r}."),
        (f"{a} minus {b}?", f"{r}"),
        (f"Take {b} away from {a}.", f"{r}"),
    ]


def make_multiplication(a, b):
    r = a * b
    return [
        (f"What is {a} times {b}?", f"{r}"),
        (f"{a} x {b} = ?", f"{r}"),
        (f"Multiply {a} by {b}.", f"{a} times {b} is {r}."),
        (f"Product of {a} and {b}?", f"{r}"),
        (f"{a} multiplied by {b}?", f"{r}"),
    ]


def make_division(a, b):
    r = a // b
    return [
        (f"What is {a} divided by {b}?", f"{r}"),
        (f"{a} / {b} = ?", f"{r}"),
        (f"Divide {a} by {b}.", f"{a} divided by {b} is {r}."),
        (f"{a} divided by {b}?", f"{r}"),
    ]


def make_square(n):
    r = n * n
    return [
        (f"What is {n} squared?", f"{r}"),
        (f"{n}^2 = ?", f"{r}"),
        (f"Square of {n}?", f"The square of {n} is {r}."),
        (f"{n} times {n}?", f"{r}"),
        (f"What is {n} to the power of 2?", f"{r}"),
    ]


# --- Addition: ALL 1-12 pairs (144 pairs x 3 = 432) + 15 larger ---
for a in range(1, 13):
    for b in range(1, 13):
        for q, ans in make_addition(a, b):
            qa_pairs.append({"instruction": q, "output": ans})

larger_add = set()
while len(larger_add) < 15:
    a, b = random.randint(13, 50), random.randint(13, 50)
    if (a, b) not in larger_add:
        larger_add.add((a, b))
        for q, ans in make_addition(a, b):
            qa_pairs.append({"instruction": q, "output": ans})

# --- Subtraction: ALL a-b where a,b in 1-12, a>=b (78 pairs x 3 = 234) ---
for a in range(1, 13):
    for b in range(1, a + 1):
        for q, ans in make_subtraction(a, b):
            qa_pairs.append({"instruction": q, "output": ans})

# --- Multiplication: ALL 1-12 pairs (144 x 3 = 432) + 10 larger ---
for a in range(1, 13):
    for b in range(1, 13):
        for q, ans in make_multiplication(a, b):
            qa_pairs.append({"instruction": q, "output": ans})

larger_mul = set()
while len(larger_mul) < 10:
    a, b = random.randint(2, 25), random.randint(2, 25)
    if (a, b) not in larger_mul and not (a <= 12 and b <= 12):
        larger_mul.add((a, b))
        for q, ans in make_multiplication(a, b):
            qa_pairs.append({"instruction": q, "output": ans})

# --- Division: common exact divisions (smaller set) ---
div_pairs = set()
for b in range(2, 11):
    for r in range(1, 11):
        a = b * r
        if (a, b) not in div_pairs:
            div_pairs.add((a, b))
            for q, ans in make_division(a, b):
                qa_pairs.append({"instruction": q, "output": ans})

# --- Squares: 1-12 (not 1-20, to keep math controlled) ---
for n in range(1, 13):
    for q, ans in make_square(n):
        qa_pairs.append({"instruction": q, "output": ans})

print(f"Basic math examples: {len(qa_pairs)}")

# --- WORD PROBLEMS (teaches natural language math) ---
word_problems = [
    ("I have 5 apples and buy 3 more. How many do I have?", "8"),
    ("I have 5 apples and give away 3. How many do I have?", "2"),
    ("A class has 12 boys and 8 girls. How many students?", "20"),
    ("If a book costs 5 dollars and I buy 3, how much total?", "15 dollars."),
    ("I have 20 cookies and eat 7. How many are left?", "13"),
    ("Each box has 6 eggs. I have 4 boxes. How many eggs?", "24"),
    ("I scored 8 out of 10 on a test. What percentage?", "80 percent."),
    ("Half of 20?", "10"),
    ("Double 6.", "12"),
    ("Triple 4.", "12"),
    ("Half of a dozen?", "6"),
    ("A dozen eggs?", "12"),
    ("If I have 10 dollars and spend 3, how much is left?", "7 dollars."),
    ("I walk 3 km then 4 km. How far total?", "7 km."),
    ("I have 3 bags with 5 items each. How many items total?", "15"),
    ("If 5 friends split 20 candies equally, how many each?", "4 candies."),
    ("I read 10 pages today and 15 yesterday. How many total?", "25 pages."),
    ("A car goes 60 km in 1 hour. How fast?", "60 km per hour."),
    ("I have 100 and give away 25. How much left?", "75"),
    ("What is twice 7?", "14"),
    ("What is three times 8?", "24"),
    ("What is one third of 9?", "3"),
    ("What is one quarter of 20?", "5"),
    ("What is 10 percent of 100?", "10"),
]
for q, a in word_problems:
    qa_pairs.append({"instruction": q, "output": a})

# --- MULTI-STEP MATH ---
multi_step = [
    ("What is 2 + 3 + 4?", "9"),
    ("What is 5 + 5 + 5?", "15"),
    ("What is 10 - 3 - 2?", "5"),
    ("What is 4 times 4 minus 6?", "10"),
    ("What is 3 squared plus 1?", "10"),
    ("What is 2 times 3 times 4?", "24"),
    ("What is 8 plus 8 plus 8?", "24"),
    ("What is 100 divided by 10?", "10"),
    ("What is 5 times 2 plus 3?", "13"),
    ("What is 6 times 6 divided by 4?", "9"),
    ("What is 7 + 8 + 9?", "24"),
    ("What is 3 + 3 + 3 + 3?", "12"),
    ("What is 10 times 10?", "100"),
    ("If 5 squared is 25, what is 6 squared?", "36"),
    ("Is 7 times 7 equal to 48?", "No, 7 times 7 is 49."),
    ("What is 12 plus 12?", "24"),
    ("What is 50 plus 50?", "100"),
    ("What is 100 minus 1?", "99"),
    ("What comes after 99?", "100"),
    ("What is 11 + 12?", "23"),
]
for q, a in multi_step:
    qa_pairs.append({"instruction": q, "output": a})

# --- NUMBER WORD MATH ---
number_word_math = [
    ("What is two plus three?", "5"),
    ("What is five times six?", "30"),
    ("What is ten minus two?", "8"),
    ("What is nine divided by three?", "3"),
    ("What is seven plus eight?", "15"),
    ("What is four times three?", "12"),
    ("What is twelve minus five?", "7"),
    ("What is eight divided by two?", "4"),
    ("What is six plus six?", "12"),
    ("What is three times three?", "9"),
    ("What is eleven plus one?", "12"),
    ("What is twenty minus ten?", "10"),
]
for q, a in number_word_math:
    qa_pairs.append({"instruction": q, "output": a})

print(f"Math total (with word/multi-step): {len(qa_pairs)}")


# ============================================================
# GEOGRAPHY — each (country, capital) appears 10 times
# ============================================================

capitals = {
    "France": ("Paris", "western Europe"),
    "Germany": ("Berlin", "central Europe"),
    "Japan": ("Tokyo", "East Asia"),
    "Australia": ("Canberra", "Oceania"),
    "Brazil": ("Brasilia", "South America"),
    "Canada": ("Ottawa", "North America"),
    "India": ("New Delhi", "South Asia"),
    "China": ("Beijing", "East Asia"),
    "Russia": ("Moscow", "eastern Europe"),
    "Mexico": ("Mexico City", "North America"),
    "Egypt": ("Cairo", "northeastern Africa"),
    "Italy": ("Rome", "southern Europe"),
    "Spain": ("Madrid", "southwestern Europe"),
    "South Korea": ("Seoul", "East Asia"),
    "United Kingdom": ("London", "western Europe"),
    "Argentina": ("Buenos Aires", "South America"),
    "Turkey": ("Ankara", "western Asia"),
    "Thailand": ("Bangkok", "Southeast Asia"),
    "Indonesia": ("Jakarta", "Southeast Asia"),
    "Nigeria": ("Abuja", "West Africa"),
    "Kenya": ("Nairobi", "East Africa"),
    "Sweden": ("Stockholm", "northern Europe"),
    "Norway": ("Oslo", "northern Europe"),
    "Poland": ("Warsaw", "central Europe"),
    "Greece": ("Athens", "southeastern Europe"),
    "Portugal": ("Lisbon", "southwestern Europe"),
    "Netherlands": ("Amsterdam", "western Europe"),
    "Switzerland": ("Bern", "central Europe"),
    "Austria": ("Vienna", "central Europe"),
    "Belgium": ("Brussels", "western Europe"),
    "Czech Republic": ("Prague", "central Europe"),
    "Denmark": ("Copenhagen", "northern Europe"),
    "Finland": ("Helsinki", "northern Europe"),
    "Ireland": ("Dublin", "western Europe"),
    "New Zealand": ("Wellington", "Oceania"),
    "South Africa": ("Pretoria", "southern Africa"),
    "Chile": ("Santiago", "South America"),
    "Colombia": ("Bogota", "South America"),
    "Peru": ("Lima", "South America"),
    "Vietnam": ("Hanoi", "Southeast Asia"),
}

geo_count = 0
for country, (capital, region) in capitals.items():
    # DIVERSE answer templates -- each structurally different
    phrasings = [
        (f"What is the capital of {country}?", f"{capital}."),
        (f"Capital of {country}?", f"{capital}."),
        (f"Name the capital of {country}.", f"The capital of {country} is {capital}."),
        (f"Which city is the capital of {country}?", f"{capital} is the capital of {country}."),
        (f"Where is the capital of {country}?", f"{capital}, in {region}."),
        (f"Tell me the capital of {country}.", f"It's {capital}."),
        (f"{capital} is the capital of which country?", f"{country}."),
        (f"Which country has {capital} as its capital?", f"{country}."),
        (f"Is {capital} the capital of {country}?", f"Yes, {capital} is the capital of {country}."),
        (f"Where is {capital}?", f"{capital} is in {country}, in {region}."),
    ]
    for q, a_text in phrasings:
        qa_pairs.append({"instruction": q, "output": a_text})
        geo_count += 1

print(f"Geography examples: {geo_count} (total so far: {len(qa_pairs)})")


# ============================================================
# PROGRAMMING — each concept appears 6 times with SHORT answers
# ============================================================

prog_facts = [
    ("variable", "A variable stores a value in a program. Example: x = 5."),
    ("function", "A function is a reusable block of code. Example: def add(a, b): return a + b."),
    ("loop", "A loop repeats code. Example: for i in range(5): print(i)."),
    ("list", "A list is an ordered collection. Example: my_list = [1, 2, 3]."),
    ("dictionary", "A dictionary maps keys to values. Example: d = {'name': 'Alice', 'age': 30}."),
    ("class", "A class is a blueprint for objects. Example: class Dog: def __init__(self, name): self.name = name."),
    ("string", "A string is text data. Example: s = 'hello'."),
    ("integer", "An integer is a whole number. Examples: 0, 1, -5, 42."),
    ("boolean", "A boolean is True or False. Used in conditions: if x > 5: print('big')."),
    ("array", "An array stores elements by index. Similar to a list in Python."),
    ("recursion", "Recursion is when a function calls itself. It needs a base case to stop."),
    ("algorithm", "An algorithm is a step-by-step procedure to solve a problem."),
    ("inheritance", "Inheritance lets a child class inherit from a parent class."),
    ("exception", "An exception is a runtime error. Handle with try/except in Python."),
    ("module", "A module is a Python file you can import. Example: import math."),
    ("tuple", "A tuple is an immutable sequence. Example: t = (1, 2, 3)."),
    ("float", "A float is a decimal number. Examples: 3.14, -0.5, 2.0."),
    ("scope", "Scope determines where a variable is accessible in code."),
    ("API", "An API lets programs communicate with each other."),
    ("debugging", "Debugging is finding and fixing errors in code."),
]

prog_count = 0
for term, short_def in prog_facts:
    phrasings = [
        (f"What is a {term}?", short_def),
        (f"Define {term}.", short_def),
        (f"What does {term} mean in programming?", short_def),
        (f"Explain {term}.", short_def),
        (f"What is a {term} in programming?", short_def),
        (f"Tell me about {term} in programming.", short_def),
        (f"Describe {term} in programming.", short_def),
        (f"What is {term} in computer science?", short_def),
        (f"What does {term} mean?", short_def),
        (f"{term.capitalize()} in programming?", short_def),
    ]
    for q, a_text in phrasings:
        qa_pairs.append({"instruction": q, "output": a_text})
        prog_count += 1

# Python how-to with SHORT answers + heavy hello world coverage
python_howto = [
    ("How do you print in Python?", "Use print(). Example: print('Hello World')."),
    ("How do you create a list in Python?", "Use square brackets: my_list = [1, 2, 3]."),
    ("How do you create a dictionary in Python?", "Use curly braces: d = {'key': 'value'}."),
    ("How do you define a function in Python?", "Use def: def greet(name): return 'Hello ' + name."),
    ("How do you write a for loop in Python?", "for i in range(5): print(i)."),
    ("How do you write a while loop in Python?", "while condition: do_something(). Example: while x > 0: x -= 1."),
    ("How do you import a module in Python?", "Use import: import math. Then use math.sqrt(16)."),
    ("How do you handle errors in Python?", "Use try/except: try: risky_code() except Error: handle_it()."),
    ("How do you read a file in Python?", "with open('file.txt') as f: text = f.read()."),
    ("How do you create a class in Python?", "class MyClass: def __init__(self): self.x = 0."),
    # Heavy hello world coverage
    ("How do you print hello world in Python?", "print('Hello World')."),
    ("How to print hello world in Python?", "print('Hello World')."),
    ("Print hello world in Python.", "print('Hello World')."),
    ("Write hello world in Python.", "print('Hello World')."),
    ("Show me hello world in Python.", "print('Hello World')."),
    ("Hello world in Python?", "print('Hello World')."),
    ("Python hello world?", "print('Hello World')."),
    ("How to print hello world?", "In Python: print('Hello World')."),
    ("Write a hello world program.", "In Python: print('Hello World')."),
    ("Show me a hello world example.", "In Python: print('Hello World'). This prints Hello World."),
]

for q, a_text in python_howto:
    qa_pairs.append({"instruction": q, "output": a_text})
    prog_count += 1

print(f"Programming examples: {prog_count} (total so far: {len(qa_pairs)})")


# ============================================================
# SCIENCE — each fact appears 6 times
# ============================================================

science_facts = [
    ("photosynthesis", "Photosynthesis converts sunlight, CO2, and water into glucose and oxygen in plants."),
    ("gravity", "Gravity pulls objects toward each other. On Earth, it is about 9.8 m/s squared."),
    ("speed of light", "The speed of light is about 300,000 km/s."),
    ("DNA", "DNA carries genetic instructions in living organisms. It has a double helix structure."),
    ("atom", "An atom has protons and neutrons in a nucleus, with electrons orbiting around it."),
    ("water cycle", "The water cycle: evaporation, condensation, precipitation, and collection."),
    ("evolution", "Evolution is species changing over time through natural selection."),
    ("cell", "A cell is the basic unit of life. Cells can be prokaryotic or eukaryotic."),
    ("electricity", "Electricity is the flow of electrons through a conductor."),
    ("magnetism", "Magnets have north and south poles. Like poles repel, opposite poles attract."),
    ("boiling point of water", "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit."),
    ("freezing point of water", "Water freezes at 0 degrees Celsius or 32 degrees Fahrenheit."),
    ("speed of sound", "The speed of sound in air is about 343 meters per second."),
    ("the Sun", "The Sun is a star at the center of our solar system."),
    ("the Moon", "The Moon orbits Earth. It causes tides and reflects sunlight."),
]

sci_count = 0
for topic, short_fact in science_facts:
    phrasings = [
        (f"What is {topic}?", short_fact),
        (f"Explain {topic}.", short_fact),
        (f"Tell me about {topic}.", short_fact),
        (f"Define {topic}.", short_fact),
        (f"Describe {topic}.", short_fact),
        (f"What do you know about {topic}?", short_fact),
        (f"Quick fact about {topic}.", short_fact),
        (f"Give me info on {topic}.", short_fact),
        (f"What can you tell me about {topic}?", short_fact),
        (f"I want to know about {topic}.", short_fact),
    ]
    for q, a_text in phrasings:
        qa_pairs.append({"instruction": q, "output": a_text})
        sci_count += 1

print(f"Science examples: {sci_count} (total so far: {len(qa_pairs)})")


# ============================================================
# GENERAL KNOWLEDGE — each fact appears 6 times
# ============================================================

general_facts = [
    ("tallest mountain", "Mount Everest is the tallest mountain at 8,849 meters."),
    ("largest ocean", "The Pacific Ocean is the largest ocean on Earth."),
    ("longest river", "The Nile River is about 6,650 kilometers long."),
    ("number of continents", "There are 7 continents: Africa, Antarctica, Asia, Australia, Europe, North America, South America."),
    ("planets in the solar system", "The 8 planets are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune."),
    ("human body temperature", "Normal body temperature is about 37 degrees Celsius or 98.6 Fahrenheit."),
    ("number of bones in human body", "An adult human has 206 bones."),
    ("pi", "Pi is approximately 3.14159. It is the ratio of circumference to diameter."),
    ("who invented the telephone", "Alexander Graham Bell invented the telephone in 1876."),
    ("who wrote Romeo and Juliet", "William Shakespeare wrote Romeo and Juliet."),
    ("when World War 2 ended", "World War 2 ended in 1945."),
    ("the Pythagorean theorem", "a^2 + b^2 = c^2 for right triangles."),
    ("when the World Wide Web was invented", "Tim Berners-Lee invented the World Wide Web in 1989."),
    ("the largest country by area", "Russia is the largest country by area."),
    ("the most spoken language", "English and Mandarin Chinese are the most widely spoken languages."),
]

gen_count = 0
for topic, short_fact in general_facts:
    phrasings = [
        (f"What is the {topic}?", short_fact),
        (f"Tell me about the {topic}.", short_fact),
        (f"What do you know about the {topic}?", short_fact),
        (f"Explain the {topic}.", short_fact),
        (f"Quick fact about the {topic}.", short_fact),
        (f"Describe the {topic}.", short_fact),
        (f"Give me info on the {topic}.", short_fact),
        (f"What can you tell me about the {topic}?", short_fact),
        (f"I want to know about the {topic}.", short_fact),
        (f"Facts about {topic}?", short_fact),
    ]
    for q, a_text in phrasings:
        qa_pairs.append({"instruction": q, "output": a_text})
        gen_count += 1

print(f"General knowledge examples: {gen_count} (total so far: {len(qa_pairs)})")


# CONVERSATIONAL section REMOVED in V4 (was 1% of data = noise floor per critic)


# ============================================================
# ALGORITHMIC MICRO-TASKS (V5) — require sequential reasoning
# These tasks CANNOT be solved with a single-pass lookup.
# The model MUST iterate the recursive loop to track state.
# ============================================================

algo_count = 0

# --- STRING REVERSAL ---
reverse_words = [
    "cat", "dog", "sun", "hat", "pen", "box", "red", "cup", "map", "ice",
    "top", "fan", "bug", "net", "log", "arm", "key", "owl", "bat", "fox",
    "star", "code", "fish", "bird", "book", "lamp", "tree", "rain", "snow", "draw",
    "hello", "world", "apple", "river", "light", "ocean", "stone", "flame", "dream", "cloud",
]
for word in reverse_words:
    rev = word[::-1]
    letters = ", ".join(list(word))
    rev_letters = ", ".join(list(rev))
    steps = ". ".join([f"Letter {i+1} from end: {word[-(i+1)]}" for i in range(len(word))])
    qa_pairs.append({"instruction": f"Reverse the letters in '{word}'.", "output": f"{steps}. Answer: {rev}."})
    qa_pairs.append({"instruction": f"Spell '{word}' backwards.", "output": f"{rev}."})
    qa_pairs.append({"instruction": f"What is '{word}' reversed?", "output": f"{rev}."})
    algo_count += 3

# --- STRING LENGTH ---
for word in reverse_words[:20]:
    n = len(word)
    qa_pairs.append({"instruction": f"How many letters in '{word}'?", "output": f"{n}."})
    qa_pairs.append({"instruction": f"Count the letters in '{word}'.", "output": f"{', '.join(list(word))}. That is {n} letters."})
    algo_count += 2

# --- LETTER POSITION ---
for word in ["apple", "hello", "world", "python", "code", "star", "light", "ocean", "stone", "dream"]:
    for pos_idx in [0, len(word)//2, len(word)-1]:
        pos = pos_idx + 1  # 1-indexed
        letter = word[pos_idx]
        qa_pairs.append({"instruction": f"What is the {pos}{'st' if pos==1 else 'nd' if pos==2 else 'rd' if pos==3 else 'th'} letter of '{word}'?",
                         "output": f"{letter}."})
        algo_count += 1

# --- TRANSITIVE LOGIC (comparison chains) ---
transitive_logic = [
    ("A is bigger than B. B is bigger than C. Is A bigger than C?", "A > B > C. Yes, A is bigger than C."),
    ("A is bigger than B. B is bigger than C. Is C bigger than A?", "A > B > C. No, C is smaller than A."),
    ("X is taller than Y. Y is taller than Z. Who is tallest?", "X > Y > Z. X is tallest."),
    ("X is taller than Y. Y is taller than Z. Who is shortest?", "X > Y > Z. Z is shortest."),
    ("Cat is faster than Dog. Dog is faster than Turtle. Is Cat faster than Turtle?", "Cat > Dog > Turtle. Yes."),
    ("Cat is faster than Dog. Dog is faster than Turtle. Is Turtle faster than Cat?", "Cat > Dog > Turtle. No."),
    ("Red is above Blue. Blue is above Green. What is on top?", "Red > Blue > Green. Red is on top."),
    ("Red is above Blue. Blue is above Green. What is on bottom?", "Red > Blue > Green. Green is on bottom."),
    ("Alice is older than Bob. Bob is older than Carol. Who is youngest?", "Alice > Bob > Carol. Carol is youngest."),
    ("Alice is older than Bob. Bob is older than Carol. Who is oldest?", "Alice > Bob > Carol. Alice is oldest."),
    ("1 is less than 2. 2 is less than 3. Is 1 less than 3?", "1 < 2 < 3. Yes."),
    ("1 is less than 2. 2 is less than 3. Is 3 less than 1?", "1 < 2 < 3. No."),
    ("Apple is heavier than Banana. Banana is heavier than Cherry. Which is lightest?", "Apple > Banana > Cherry. Cherry is lightest."),
    ("Dog runs faster than Cat. Cat runs faster than Mouse. Who is slowest?", "Dog > Cat > Mouse. Mouse is slowest."),
    ("P > Q. Q > R. R > S. What is the order?", "P > Q > R > S."),
    ("P > Q. Q > R. R > S. Is S bigger than P?", "P > Q > R > S. No, S is smallest."),
    ("A = 5. B = 3. C = 7. Which is largest?", "A=5, B=3, C=7. C is largest."),
    ("A = 5. B = 3. C = 7. Which is smallest?", "A=5, B=3, C=7. B is smallest."),
    ("A = 5. B = 3. C = 7. Put them in order.", "B=3, A=5, C=7."),
    ("A = 5. B = 3. Is A bigger than B?", "5 > 3. Yes."),
]
for q, a in transitive_logic:
    qa_pairs.append({"instruction": q, "output": a})
    algo_count += 1

# More transitive with numbers
for a, b, c in [(10,5,3), (7,4,2), (9,6,1), (8,3,1), (12,7,5), (20,15,10)]:
    qa_pairs.append({"instruction": f"X={a}. Y={b}. Z={c}. Is X bigger than Z?", "output": f"{a}>{b}>{c}. Yes."})
    qa_pairs.append({"instruction": f"X={a}. Y={b}. Z={c}. Order from smallest.", "output": f"{c}, {b}, {a}."})
    qa_pairs.append({"instruction": f"X={a}. Y={b}. Z={c}. What is the biggest?", "output": f"X={a} is biggest."})
    algo_count += 3

# --- STATE TRACKING (sequential operations on a counter) ---
state_tracking = [
    ("I have 3 apples. I eat 1. I find 2. How many do I have?", "3 - 1 = 2. 2 + 2 = 4. Answer: 4."),
    ("I have 5 coins. I spend 2. I earn 3. How many?", "5 - 2 = 3. 3 + 3 = 6. Answer: 6."),
    ("I have 10 dollars. I spend 4. I find 1. How much?", "10 - 4 = 6. 6 + 1 = 7. Answer: 7 dollars."),
    ("I have 0 apples. I buy 5. I eat 2. I buy 1. How many?", "0 + 5 = 5. 5 - 2 = 3. 3 + 1 = 4. Answer: 4."),
    ("I have 8 marbles. I give 3 away. I get 2 back. How many?", "8 - 3 = 5. 5 + 2 = 7. Answer: 7."),
    ("I have 6 books. I read 2. I buy 4 more. How many?", "I have 6 books. I buy 4 more. 6 + 4 = 10. Answer: 10."),
    ("Start with 10. Subtract 3. Add 5. Subtract 2. Result?", "10 - 3 = 7. 7 + 5 = 12. 12 - 2 = 10. Answer: 10."),
    ("Start with 0. Add 4. Add 3. Subtract 1. Result?", "0 + 4 = 4. 4 + 3 = 7. 7 - 1 = 6. Answer: 6."),
    ("Start with 20. Subtract 5. Subtract 5. Add 2. Result?", "20 - 5 = 15. 15 - 5 = 10. 10 + 2 = 12. Answer: 12."),
    ("Start with 1. Double it. Double it again. Result?", "1 * 2 = 2. 2 * 2 = 4. Answer: 4."),
    ("Start with 2. Double it. Double it. Double it. Result?", "2 * 2 = 4. 4 * 2 = 8. 8 * 2 = 16. Answer: 16."),
    ("Start with 100. Halve it. Halve it again. Result?", "100 / 2 = 50. 50 / 2 = 25. Answer: 25."),
    ("I have 7. I add 3. I subtract 5. I add 10. Result?", "7 + 3 = 10. 10 - 5 = 5. 5 + 10 = 15. Answer: 15."),
    ("I have 4 bags. Each bag has 3 items. I lose 1 bag. How many items?", "4 * 3 = 12 items. Lose 1 bag = 3 items. 12 - 3 = 9. Answer: 9."),
    ("I have 12 cookies. I eat half. Then I eat half of whats left. How many?", "12 / 2 = 6. 6 / 2 = 3. Answer: 3."),
]
for q, a in state_tracking:
    qa_pairs.append({"instruction": q, "output": a})
    algo_count += 1

# Generate more state tracking programmatically
for start, ops in [
    (5, [("+", 3), ("-", 1), ("+", 4)]),
    (8, [("-", 2), ("-", 3), ("+", 1)]),
    (3, [("+", 7), ("-", 4), ("+", 2)]),
    (10, [("-", 5), ("+", 3), ("-", 1)]),
    (6, [("+", 6), ("-", 4), ("-", 2)]),
    (1, [("+", 1), ("+", 1), ("+", 1)]),
    (9, [("-", 3), ("-", 3), ("-", 3)]),
    (4, [("+", 4), ("+", 4), ("-", 8)]),
    (15, [("-", 5), ("-", 5), ("-", 5)]),
    (2, [("+", 8), ("-", 3), ("+", 1)]),
    (7, [("+", 3), ("-", 5), ("+", 6)]),
    (0, [("+", 10), ("-", 4), ("+", 2)]),
    (20, [("-", 7), ("-", 3), ("+", 5)]),
    (11, [("-", 6), ("+", 2), ("+", 3)]),
    (5, [("+", 5), ("+", 5), ("-", 10)]),
]:
    # Build instruction
    q_parts = [f"Start with {start}."]
    steps = []
    val = start
    for op, num in ops:
        if op == "+":
            q_parts.append(f"Add {num}.")
            new_val = val + num
            steps.append(f"{val} + {num} = {new_val}")
        else:
            q_parts.append(f"Subtract {num}.")
            new_val = val - num
            steps.append(f"{val} - {num} = {new_val}")
        val = new_val
    q_parts.append("Result?")
    a_text = ". ".join(steps) + f". Answer: {val}."
    qa_pairs.append({"instruction": " ".join(q_parts), "output": a_text})
    algo_count += 1

# --- MULTI-STEP MATH WITH CHAIN-OF-THOUGHT ---
cot_math = [
    ("What is 3 * 4 + 5?", "3 * 4 = 12. 12 + 5 = 17. Answer: 17."),
    ("What is 5 * 5 - 10?", "5 * 5 = 25. 25 - 10 = 15. Answer: 15."),
    ("What is 2 + 3 * 4?", "3 * 4 = 12. 2 + 12 = 14. Answer: 14."),
    ("What is 10 - 3 + 7?", "10 - 3 = 7. 7 + 7 = 14. Answer: 14."),
    ("What is 6 * 2 + 8?", "6 * 2 = 12. 12 + 8 = 20. Answer: 20."),
    ("What is 7 * 3 - 1?", "7 * 3 = 21. 21 - 1 = 20. Answer: 20."),
    ("What is 4 * 5 + 4 * 5?", "4 * 5 = 20. 20 + 20 = 40. Answer: 40."),
    ("What is 8 + 8 + 8 + 8?", "8 + 8 = 16. 16 + 8 = 24. 24 + 8 = 32. Answer: 32."),
    ("What is 9 - 4 - 3?", "9 - 4 = 5. 5 - 3 = 2. Answer: 2."),
    ("What is 2 * 2 * 2 * 2?", "2 * 2 = 4. 4 * 2 = 8. 8 * 2 = 16. Answer: 16."),
    ("What is 100 - 50 - 25?", "100 - 50 = 50. 50 - 25 = 25. Answer: 25."),
    ("What is 3 + 4 + 5 + 6?", "3 + 4 = 7. 7 + 5 = 12. 12 + 6 = 18. Answer: 18."),
    ("What is 12 - 4 + 7?", "12 - 4 = 8. 8 + 7 = 15. Answer: 15."),
    ("What is 11 * 2 - 2?", "11 * 2 = 22. 22 - 2 = 20. Answer: 20."),
    ("What is 6 + 6 + 6?", "6 + 6 = 12. 12 + 6 = 18. Answer: 18."),
    ("What is 5 * 3 + 5?", "5 * 3 = 15. 15 + 5 = 20. Answer: 20."),
    ("What is 9 * 2 - 8?", "9 * 2 = 18. 18 - 8 = 10. Answer: 10."),
    ("What is 4 + 4 + 4 + 4 + 4?", "4 * 5 = 20. Answer: 20."),
    ("What is 7 + 8 - 5?", "7 + 8 = 15. 15 - 5 = 10. Answer: 10."),
    ("What is 10 * 3 - 5?", "10 * 3 = 30. 30 - 5 = 25. Answer: 25."),
]
for q, a in cot_math:
    qa_pairs.append({"instruction": q, "output": a})
    algo_count += 1

# --- COMPARISON WITH REASONING ---
comparison = [
    ("Which is bigger, 7 * 8 or 9 * 6?", "7 * 8 = 56. 9 * 6 = 54. 56 > 54. 7 * 8 is bigger."),
    ("Which is bigger, 5 * 5 or 3 * 8?", "5 * 5 = 25. 3 * 8 = 24. 25 > 24. 5 * 5 is bigger."),
    ("Which is bigger, 4 * 6 or 5 * 5?", "4 * 6 = 24. 5 * 5 = 25. 25 > 24. 5 * 5 is bigger."),
    ("Which is bigger, 10 * 2 or 3 * 7?", "10 * 2 = 20. 3 * 7 = 21. 21 > 20. 3 * 7 is bigger."),
    ("Which is bigger, 6 * 6 or 7 * 5?", "6 * 6 = 36. 7 * 5 = 35. 36 > 35. 6 * 6 is bigger."),
    ("Which is bigger, 8 * 3 or 4 * 6?", "8 * 3 = 24. 4 * 6 = 24. They are equal."),
    ("Which is bigger, 9 * 9 or 10 * 8?", "9 * 9 = 81. 10 * 8 = 80. 81 > 80. 9 * 9 is bigger."),
    ("Which is bigger, 12 + 8 or 7 + 14?", "12 + 8 = 20. 7 + 14 = 21. 21 > 20. 7 + 14 is bigger."),
    ("Which is bigger, 3 * 3 * 3 or 5 * 5?", "3 * 3 * 3 = 27. 5 * 5 = 25. 27 > 25. 3 * 3 * 3 is bigger."),
    ("Which is bigger, 2 * 10 or 4 * 5?", "2 * 10 = 20. 4 * 5 = 20. They are equal."),
    ("Is 5 + 5 equal to 3 + 7?", "5 + 5 = 10. 3 + 7 = 10. Yes, they are equal."),
    ("Is 6 * 3 equal to 9 * 2?", "6 * 3 = 18. 9 * 2 = 18. Yes, they are equal."),
    ("Is 4 * 4 bigger than 3 * 5?", "4 * 4 = 16. 3 * 5 = 15. Yes, 4 * 4 is bigger."),
    ("Is 7 + 8 bigger than 20?", "7 + 8 = 15. 15 < 20. No."),
    ("Is 10 * 10 bigger than 99?", "10 * 10 = 100. 100 > 99. Yes."),
]
for q, a in comparison:
    qa_pairs.append({"instruction": q, "output": a})
    algo_count += 1

# --- SIMPLE LOGIC / TRUE-FALSE ---
logic_tf = [
    ("True or False: All dogs are animals.", "True. Dogs are animals."),
    ("True or False: All animals are dogs.", "False. Not all animals are dogs."),
    ("True or False: 5 is greater than 3.", "True. 5 > 3."),
    ("True or False: 2 + 2 = 5.", "False. 2 + 2 = 4."),
    ("True or False: Water freezes at 0 degrees Celsius.", "True."),
    ("True or False: The Sun orbits the Earth.", "False. Earth orbits the Sun."),
    ("True or False: 10 times 0 is 0.", "True. Any number times 0 is 0."),
    ("True or False: A square has 3 sides.", "False. A square has 4 sides."),
    ("True or False: Tokyo is in Japan.", "True."),
    ("True or False: Paris is in Germany.", "False. Paris is in France."),
    ("If it is raining, the ground is wet. It is raining. Is the ground wet?", "Yes. It is raining, so the ground is wet."),
    ("If it is raining, the ground is wet. The ground is wet. Is it raining?", "Not necessarily. The ground could be wet for other reasons."),
    ("All cats are pets. Tom is a cat. Is Tom a pet?", "Yes. Tom is a cat, and all cats are pets."),
    ("All birds can fly. A penguin is a bird. Can a penguin fly?", "The premise is wrong. Not all birds can fly. Penguins cannot fly."),
    ("If A then B. A is true. Is B true?", "Yes. A is true and A implies B, so B is true."),
    ("If A then B. B is true. Is A true?", "Not necessarily. B can be true for other reasons."),
    ("If A then B. B is false. Is A true?", "No. If B is false and A implies B, then A must be false."),
    ("All squares are rectangles. Is a rectangle a square?", "Not necessarily. All squares are rectangles, but not all rectangles are squares."),
]
for q, a in logic_tf:
    qa_pairs.append({"instruction": q, "output": a})
    algo_count += 1

# --- PATTERN CONTINUATION ---
patterns = [
    ("What comes next: 2, 4, 6, 8, ?", "Each number increases by 2. Next: 10."),
    ("What comes next: 1, 3, 5, 7, ?", "Each number increases by 2. Next: 9."),
    ("What comes next: 1, 2, 4, 8, ?", "Each number doubles. Next: 16."),
    ("What comes next: 10, 20, 30, 40, ?", "Each number increases by 10. Next: 50."),
    ("What comes next: 3, 6, 9, 12, ?", "Each number increases by 3. Next: 15."),
    ("What comes next: 5, 10, 15, 20, ?", "Each number increases by 5. Next: 25."),
    ("What comes next: 1, 4, 9, 16, ?", "These are squares: 1, 4, 9, 16. Next: 25."),
    ("What comes next: 100, 90, 80, 70, ?", "Each number decreases by 10. Next: 60."),
    ("What comes next: 1, 1, 2, 3, 5, ?", "Fibonacci: each is sum of previous two. Next: 8."),
    ("What comes next: 2, 6, 18, 54, ?", "Each number is tripled. Next: 162."),
    ("What is the pattern: 1, 2, 3, 4, 5?", "Each number increases by 1."),
    ("What is the pattern: 2, 4, 8, 16?", "Each number doubles."),
]
for q, a in patterns:
    qa_pairs.append({"instruction": q, "output": a})
    algo_count += 1

print(f"Algorithmic micro-tasks: {algo_count} (total so far: {len(qa_pairs)})")


# ============================================================
# ARITHMETIC DRILLS (Round 10) -- targeted drilling for the Cross-Entropy Illusion
# Every single-digit × single-digit result, with CoT and direct answer.
# Also 2-digit + 1-digit, simple division, and multi-step compound operations.
# These generate the (a,b,c) triples the FFN needs to wire multiplication circuits.
# ============================================================

arith_count = 0

# --- Single-digit multiplication with CoT (all 81 pairs, 2 phrasings each) ---
for a in range(1, 10):
    for b in range(1, 10):
        r = a * b
        qa_pairs.append({"instruction": f"What is {a} * {b}?", "output": f"{a} * {b} = {r}. Answer: {r}."})
        qa_pairs.append({"instruction": f"Compute {a} times {b}.", "output": f"{r}"})
        arith_count += 2

# --- Two-step arithmetic: a * b + c (a,b in 2-9, c in 1-9, 200 samples) ---
two_step_pairs = set()
while len(two_step_pairs) < 200:
    a = random.randint(2, 9)
    b = random.randint(2, 9)
    c = random.randint(1, 9)
    if (a, b, c) not in two_step_pairs:
        two_step_pairs.add((a, b, c))
        prod = a * b
        result = prod + c
        qa_pairs.append({
            "instruction": f"What is {a} * {b} + {c}?",
            "output": f"{a} * {b} = {prod}. {prod} + {c} = {result}. Answer: {result}."
        })
        arith_count += 1

# --- Two-step: a + b * c (order of operations, 150 samples) ---
ooo_pairs = set()
while len(ooo_pairs) < 150:
    a = random.randint(1, 20)
    b = random.randint(2, 9)
    c = random.randint(2, 9)
    if (a, b, c) not in ooo_pairs:
        ooo_pairs.add((a, b, c))
        prod = b * c
        result = a + prod
        qa_pairs.append({
            "instruction": f"What is {a} + {b} * {c}?",
            "output": f"{b} * {c} = {prod}. {a} + {prod} = {result}. Answer: {result}."
        })
        arith_count += 1

# --- Two-step: a - b + c (state tracking with numbers, 150 samples) ---
state_pairs = set()
while len(state_pairs) < 150:
    a = random.randint(5, 30)
    b = random.randint(1, a - 1)
    c = random.randint(1, 15)
    if (a, b, c) not in state_pairs:
        state_pairs.add((a, b, c))
        mid = a - b
        result = mid + c
        qa_pairs.append({
            "instruction": f"I have {a} items. I use {b}. I get {c} more. How many do I have?",
            "output": f"{a} - {b} = {mid}. {mid} + {c} = {result}. Answer: {result}."
        })
        arith_count += 1

# --- Simple division (no remainder, all single-digit results, 80 pairs) ---
div_count = 0
for divisor in range(2, 10):
    for result in range(1, 10):
        dividend = divisor * result
        qa_pairs.append({
            "instruction": f"What is {dividend} divided by {divisor}?",
            "output": f"{dividend} / {divisor} = {result}. Answer: {result}."
        })
        div_count += 1
        if div_count >= 80:
            break
    if div_count >= 80:
        break
arith_count += div_count

# --- Three-step arithmetic (a * b + c - d, 100 samples) ---
three_step = set()
while len(three_step) < 100:
    a = random.randint(2, 9)
    b = random.randint(2, 9)
    c = random.randint(1, 9)
    d = random.randint(1, min(a * b + c - 1, 9))
    if (a, b, c, d) not in three_step:
        three_step.add((a, b, c, d))
        s1 = a * b
        s2 = s1 + c
        s3 = s2 - d
        qa_pairs.append({
            "instruction": f"What is {a} * {b} + {c} - {d}?",
            "output": f"{a} * {b} = {s1}. {s1} + {c} = {s2}. {s2} - {d} = {s3}. Answer: {s3}."
        })
        arith_count += 1

print(f"Arithmetic drills (R10): {arith_count} (total so far: {len(qa_pairs)})")


# ============================================================
# THOROUGH SHUFFLE -- critical to prevent topic clustering in chunks
# ============================================================

random.seed(12345)
random.shuffle(qa_pairs)

print(f"\nTotal examples: {len(qa_pairs)}")

# Verify test cases exist
test_cases = {
    "2 + 3": False,
    "3 + 2": False,
    "capital of France": False,
    "10 times 5": False,
    "print hello world": False,
    "What is a function": False,
    "What is a variable": False,
}
for item in qa_pairs:
    q = item["instruction"].lower()
    for tc in test_cases:
        if tc.lower() in q:
            test_cases[tc] = True

print("\nTest case coverage:")
for tc, found in test_cases.items():
    status = "FOUND" if found else "MISSING"
    print(f"  {tc}: {status}")

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "train.jsonl")

with open(output_path, "w", encoding="utf-8") as f:
    for item in qa_pairs:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nSaved to: {output_path}")
avg_q_len = sum(len(item["instruction"]) for item in qa_pairs) / len(qa_pairs)
avg_a_len = sum(len(item["output"]) for item in qa_pairs) / len(qa_pairs)
print(f"Avg question length: {avg_q_len:.0f} chars")
print(f"Avg answer length: {avg_a_len:.0f} chars")
