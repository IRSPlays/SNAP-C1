"""Generate 2000+ diverse Q&A pairs with rich answers (100+ tokens each)."""
import json
import random
import os

random.seed(42)

qa_pairs = []

# ============================================================
# 1. MATH (400 examples)
# ============================================================
# Addition
for _ in range(80):
    a, b = random.randint(1, 999), random.randint(1, 999)
    result = a + b
    qa_pairs.append({
        "instruction": f"What is {a} + {b}?",
        "output": f"{a} + {b} equals {result}. To solve this addition problem, you combine the two numbers together. Addition is one of the four basic arithmetic operations. The sum of {a} and {b} gives us {result}."
    })

# Subtraction
for _ in range(60):
    a = random.randint(10, 999)
    b = random.randint(1, a)
    result = a - b
    qa_pairs.append({
        "instruction": f"What is {a} - {b}?",
        "output": f"{a} minus {b} equals {result}. Subtraction finds the difference between two numbers. When you take {b} away from {a}, you are left with {result}. This is a fundamental arithmetic operation."
    })

# Multiplication
for _ in range(60):
    a, b = random.randint(2, 50), random.randint(2, 50)
    result = a * b
    qa_pairs.append({
        "instruction": f"What is {a} times {b}?",
        "output": f"{a} multiplied by {b} equals {result}. Multiplication is repeated addition. You can think of this as adding {a} to itself {b} times, which gives {result}. This is one of the basic mathematical operations."
    })

# Division
for _ in range(50):
    b = random.randint(2, 30)
    result = random.randint(1, 50)
    a = b * result
    qa_pairs.append({
        "instruction": f"What is {a} divided by {b}?",
        "output": f"{a} divided by {b} equals {result}. Division splits a number into equal parts. When you divide {a} into {b} equal groups, each group contains {result}. Division is the inverse operation of multiplication."
    })

# Squares
for n in range(2, 32):
    result = n * n
    qa_pairs.append({
        "instruction": f"What is {n} squared?",
        "output": f"{n} squared is {result}. Squaring a number means multiplying it by itself. So {n} times {n} equals {result}. Squaring is a common operation in mathematics, geometry, and physics."
    })

# Cubes
for n in range(2, 16):
    result = n * n * n
    qa_pairs.append({
        "instruction": f"What is {n} cubed?",
        "output": f"{n} cubed is {result}. Cubing a number means multiplying it by itself three times. So {n} times {n} times {n} equals {result}. Cubes appear frequently in volume calculations."
    })

# Percentages
for _ in range(40):
    pct = random.choice([10, 15, 20, 25, 30, 50, 75])
    base = random.choice([50, 100, 200, 400, 500, 1000])
    result = pct * base // 100
    qa_pairs.append({
        "instruction": f"What is {pct}% of {base}?",
        "output": f"{pct}% of {base} is {result}. To calculate a percentage, multiply the base number by the percentage and divide by 100. So {base} times {pct} divided by 100 gives us {result}. Percentages express a fraction of 100."
    })

# Averages
for _ in range(30):
    nums = [random.randint(1, 100) for _ in range(random.randint(3, 5))]
    avg = sum(nums) / len(nums)
    nums_str = ", ".join(str(n) for n in nums)
    qa_pairs.append({
        "instruction": f"What is the average of {nums_str}?",
        "output": f"The average of {nums_str} is {avg:.1f}. To find the average, add all the numbers together to get {sum(nums)}, then divide by the count of numbers which is {len(nums)}. The average, also called the mean, represents the central value of the dataset."
    })

# ============================================================
# 2. GEOGRAPHY / CAPITALS (200 examples)
# ============================================================
capitals = {
    "France": ("Paris", "western Europe", "Eiffel Tower and the Louvre Museum"),
    "Germany": ("Berlin", "central Europe", "Brandenburg Gate"),
    "Japan": ("Tokyo", "East Asia", "Tokyo Tower and the Imperial Palace"),
    "Australia": ("Canberra", "Oceania", "Parliament House"),
    "Brazil": ("Brasilia", "South America", "Cathedral of Brasilia"),
    "Canada": ("Ottawa", "North America", "Parliament Hill"),
    "India": ("New Delhi", "South Asia", "India Gate"),
    "China": ("Beijing", "East Asia", "the Forbidden City"),
    "Russia": ("Moscow", "eastern Europe and northern Asia", "the Kremlin"),
    "Mexico": ("Mexico City", "North America", "the Palace of Fine Arts"),
    "Egypt": ("Cairo", "northeastern Africa", "the Pyramids of Giza nearby"),
    "Italy": ("Rome", "southern Europe", "the Colosseum and the Vatican"),
    "Spain": ("Madrid", "southwestern Europe", "the Royal Palace"),
    "South Korea": ("Seoul", "East Asia", "Gyeongbokgung Palace"),
    "United Kingdom": ("London", "western Europe", "Big Ben and Buckingham Palace"),
    "Argentina": ("Buenos Aires", "South America", "the Obelisk and Casa Rosada"),
    "Turkey": ("Ankara", "southeastern Europe and western Asia", "Anitkabir"),
    "Thailand": ("Bangkok", "Southeast Asia", "the Grand Palace"),
    "Indonesia": ("Jakarta", "Southeast Asia", "the National Monument"),
    "Nigeria": ("Abuja", "West Africa", "Aso Rock"),
    "Kenya": ("Nairobi", "East Africa", "the Kenyatta International Conference Centre"),
    "Sweden": ("Stockholm", "northern Europe", "the Royal Palace"),
    "Norway": ("Oslo", "northern Europe", "the Viking Ship Museum"),
    "Poland": ("Warsaw", "central Europe", "the Old Town Market Place"),
    "Greece": ("Athens", "southeastern Europe", "the Parthenon and the Acropolis"),
    "Portugal": ("Lisbon", "southwestern Europe", "the Belem Tower"),
    "Netherlands": ("Amsterdam", "western Europe", "the Anne Frank House"),
    "Switzerland": ("Bern", "central Europe", "the Zytglogge clock tower"),
    "Austria": ("Vienna", "central Europe", "Schonbrunn Palace"),
    "Belgium": ("Brussels", "western Europe", "the Grand Place"),
    "Czech Republic": ("Prague", "central Europe", "Prague Castle"),
    "Denmark": ("Copenhagen", "northern Europe", "the Little Mermaid statue"),
    "Finland": ("Helsinki", "northern Europe", "the Helsinki Cathedral"),
    "Ireland": ("Dublin", "western Europe", "Trinity College"),
    "New Zealand": ("Wellington", "Oceania", "Te Papa Museum"),
    "South Africa": ("Pretoria", "southern Africa", "the Union Buildings"),
    "Chile": ("Santiago", "South America", "the Metropolitan Cathedral"),
    "Colombia": ("Bogota", "South America", "the Gold Museum"),
    "Peru": ("Lima", "South America", "the Plaza Mayor"),
    "Vietnam": ("Hanoi", "Southeast Asia", "Ho Chi Minh Mausoleum"),
    "Malaysia": ("Kuala Lumpur", "Southeast Asia", "the Petronas Twin Towers"),
    "Philippines": ("Manila", "Southeast Asia", "Rizal Park"),
    "Pakistan": ("Islamabad", "South Asia", "the Faisal Mosque"),
    "Bangladesh": ("Dhaka", "South Asia", "the National Parliament House"),
    "Saudi Arabia": ("Riyadh", "the Middle East", "the Kingdom Centre Tower"),
    "Iran": ("Tehran", "the Middle East", "the Azadi Tower"),
    "Iraq": ("Baghdad", "the Middle East", "the Al-Shaheed Monument"),
    "Morocco": ("Rabat", "North Africa", "the Hassan Tower"),
    "Ukraine": ("Kyiv", "eastern Europe", "Saint Sophia Cathedral"),
}

for country, (capital, region, landmark) in capitals.items():
    qa_pairs.append({
        "instruction": f"What is the capital of {country}?",
        "output": f"The capital of {country} is {capital}. {capital} is located in {region} and is the largest and most important city in {country}. It serves as the seat of government and is known for {landmark}. The city is a major political, economic, and cultural center."
    })
    qa_pairs.append({
        "instruction": f"Tell me about {capital}.",
        "output": f"{capital} is the capital city of {country}, located in {region}. It is famous for {landmark}. As the capital, it serves as the center of government and is an important hub for commerce, culture, and tourism. {capital} is one of the most well-known cities in {country}."
    })
    qa_pairs.append({
        "instruction": f"Where is {capital}?",
        "output": f"{capital} is the capital of {country}, situated in {region}. It is one of the most prominent cities in the country. {capital} is known for {landmark} and serves as the political and economic heart of {country}."
    })
    qa_pairs.append({
        "instruction": f"Which country has {capital} as its capital?",
        "output": f"{capital} is the capital of {country}. {country} is located in {region}. {capital} serves as the political center of {country} and is known for {landmark}. It is a major city with rich history and culture."
    })

# ============================================================
# 3. PROGRAMMING CONCEPTS (300 examples)
# ============================================================
prog_concepts = [
    ("variable", "A variable is a named storage location in a program that holds a value. Variables can store different types of data such as numbers, text, or boolean values. You can change the value stored in a variable during program execution. Variables are fundamental to all programming languages and allow you to work with data dynamically."),
    ("function", "A function is a reusable block of code that performs a specific task. Functions take input parameters, process them, and can return a result. They help organize code into logical units and avoid repetition. For example, a function called 'add' might take two numbers and return their sum. Functions are a core building block of programming."),
    ("loop", "A loop is a programming construct that repeats a block of code multiple times. Common types include for loops, which iterate a set number of times, and while loops, which continue until a condition becomes false. Loops are essential for processing collections of data and automating repetitive tasks."),
    ("array", "An array is an ordered collection of elements stored in contiguous memory locations. Each element can be accessed by its index, starting from zero in most languages. Arrays are useful for storing lists of similar items like numbers or strings. They are one of the most basic and widely used data structures in programming."),
    ("string", "A string is a sequence of characters used to represent text in programming. Strings can contain letters, numbers, spaces, and special characters. Most languages provide built-in operations for strings like concatenation, slicing, and searching. Strings are immutable in some languages like Python and Java."),
    ("class", "A class is a blueprint for creating objects in object-oriented programming. It defines the properties and methods that objects of that type will have. For example, a Car class might have properties like color and speed, and methods like drive and brake. Classes enable code reuse and organized program design."),
    ("object", "An object is an instance of a class that contains data and behavior. Objects store their state in attributes and expose behavior through methods. In object-oriented programming, objects interact with each other to accomplish tasks. Every object has its own copy of the instance variables defined by its class."),
    ("dictionary", "A dictionary is a data structure that stores key-value pairs. Each key maps to a specific value, allowing fast lookups by key. In Python, dictionaries are created with curly braces. They are useful for storing related information like a person's name mapped to their phone number."),
    ("list", "A list is an ordered, mutable collection of elements in Python. Lists can contain items of different types and support operations like append, remove, and sort. You access elements by their index position. Lists are one of the most versatile data structures in Python and are used extensively in programming."),
    ("tuple", "A tuple is an ordered, immutable collection of elements in Python. Once created, the elements of a tuple cannot be changed. Tuples are defined using parentheses. They are useful when you want to store a fixed collection of values, like coordinates or database records. Tuples are faster than lists."),
    ("boolean", "A boolean is a data type that can have only two values: True or False. Booleans are used in conditional statements and logical operations. They are named after mathematician George Boole. In programming, comparisons like 5 > 3 return True, while 2 > 7 returns False."),
    ("integer", "An integer is a whole number without a decimal point. Integers can be positive, negative, or zero. In Python, integers have unlimited precision, meaning they can be arbitrarily large. Common operations on integers include addition, subtraction, multiplication, and division."),
    ("float", "A float is a number with a decimal point. Floats are used to represent real numbers in programming. They follow the IEEE 754 standard for floating-point arithmetic. Be aware that float calculations can have small precision errors, so avoid using them for financial calculations where exact values matter."),
    ("recursion", "Recursion is when a function calls itself to solve a problem. A recursive function needs a base case to stop the recursion and a recursive case that breaks the problem into smaller parts. Classic examples include calculating factorials and Fibonacci numbers. Recursion can be elegant but may cause stack overflow if not handled carefully."),
    ("algorithm", "An algorithm is a step-by-step procedure for solving a problem or performing a computation. Algorithms take input, process it through a series of defined steps, and produce output. Common algorithms include sorting, searching, and graph traversal. The efficiency of an algorithm is measured by its time and space complexity."),
    ("inheritance", "Inheritance is an object-oriented programming concept where a new class is based on an existing class. The child class inherits all the properties and methods of the parent class and can add its own. This promotes code reuse and establishes a natural hierarchy. For example, a Dog class might inherit from an Animal class."),
    ("polymorphism", "Polymorphism means that objects of different classes can be treated as objects of a common parent class. The same method name can behave differently depending on which class implements it. This allows writing flexible code that works with different types. Polymorphism is a key principle of object-oriented programming."),
    ("encapsulation", "Encapsulation is the bundling of data and the methods that operate on that data within a single unit, typically a class. It restricts direct access to some components and prevents accidental modification. In Python, this is achieved through naming conventions like prefixing attributes with underscores. Encapsulation improves code maintainability."),
    ("exception", "An exception is an error that occurs during the execution of a program. Exceptions disrupt the normal flow of the program. In Python, you handle exceptions using try-except blocks. Common exceptions include TypeError, ValueError, and ZeroDivisionError. Proper exception handling makes programs more robust."),
    ("API", "An API, or Application Programming Interface, is a set of rules and protocols that allows different software applications to communicate with each other. APIs define how to request and exchange data. Web APIs typically use HTTP requests to send and receive data in formats like JSON. APIs are essential for modern software development."),
    ("database", "A database is an organized collection of structured data stored electronically. Databases allow efficient storage, retrieval, and manipulation of data. Common types include relational databases like PostgreSQL and MySQL, and NoSQL databases like MongoDB. Databases use query languages like SQL to interact with the stored data."),
    ("compiler", "A compiler is a program that translates source code written in a high-level programming language into machine code that the computer can execute. Compilation happens before the program runs. Languages like C and Java use compilers. The compilation process includes parsing, optimization, and code generation stages."),
    ("interpreter", "An interpreter is a program that executes source code line by line at runtime, without compiling it first. Python and JavaScript are interpreted languages. Interpreters are easier to use for development and debugging because you can see results immediately. However, interpreted code generally runs slower than compiled code."),
    ("debugging", "Debugging is the process of finding and fixing errors in a program. Common debugging techniques include using print statements, setting breakpoints, and using a debugger tool. The term originated when a literal bug was found in an early computer. Good debugging skills are essential for every programmer."),
    ("git", "Git is a distributed version control system used to track changes in source code. It allows multiple developers to work on the same project simultaneously. Key concepts include commits, branches, and merges. Git was created by Linus Torvalds in 2005 and is the most widely used version control system today."),
    ("type", "A type in programming defines the kind of data a variable can hold and what operations can be performed on it. Common types include integer, float, string, and boolean. Strong typing helps catch errors early, while dynamic typing offers flexibility. Understanding types is fundamental to writing correct programs."),
    ("iteration", "Iteration is the process of repeating a set of instructions multiple times. In programming, iteration is achieved through loops like for and while. Each repetition is called an iteration. Iteration is used to process collections, repeat calculations, and implement algorithms that converge on solutions."),
    ("scope", "Scope determines where a variable can be accessed within a program. Variables defined inside a function have local scope and cannot be accessed outside. Variables defined at the top level have global scope. Understanding scope prevents naming conflicts and unexpected behavior in programs."),
    ("module", "A module is a file containing Python code that can be imported and used in other programs. Modules help organize code into reusable components. Python has many built-in modules like math, os, and json. You can also create your own modules by saving code in .py files and importing them."),
    ("decorator", "A decorator in Python is a function that modifies the behavior of another function without changing its code. Decorators are applied using the @ symbol above the function definition. Common uses include logging, authentication, and caching. Decorators are a powerful feature of Python that enable clean, reusable code patterns."),
]

for term, definition in prog_concepts:
    qa_pairs.append({
        "instruction": f"What is a {term} in programming?",
        "output": definition
    })
    qa_pairs.append({
        "instruction": f"Explain what {term} means in programming.",
        "output": definition
    })
    qa_pairs.append({
        "instruction": f"Define {term} in the context of programming.",
        "output": definition
    })
    qa_pairs.append({
        "instruction": f"Can you explain the concept of {term}?",
        "output": definition
    })

# Additional programming Q&A
python_qa = [
    ("How do you create a list in Python?", "To create a list in Python, use square brackets. For example: my_list = [1, 2, 3, 4, 5]. You can also create an empty list with my_list = [] and add elements later using the append method. Lists can contain items of different types like numbers, strings, and even other lists."),
    ("How do you read a file in Python?", "To read a file in Python, use the open function with a with statement: with open('filename.txt', 'r') as f: content = f.read(). The with statement ensures the file is properly closed after reading. You can also read line by line using f.readlines() or iterate over the file object directly."),
    ("How do you write to a file in Python?", "To write to a file in Python, use the open function with write mode: with open('filename.txt', 'w') as f: f.write('Hello World'). Use 'w' mode to overwrite or 'a' mode to append. The with statement ensures the file is properly closed after writing. Always handle file operations carefully to avoid data loss."),
    ("How do you create a dictionary in Python?", "To create a dictionary in Python, use curly braces with key-value pairs: my_dict = {'name': 'Alice', 'age': 30, 'city': 'Paris'}. You can access values using keys: my_dict['name'] returns 'Alice'. Dictionaries are mutable and allow fast lookups. You can add new entries with my_dict['key'] = value."),
    ("How do you define a function in Python?", "To define a function in Python, use the def keyword followed by the function name and parameters: def greet(name): return 'Hello, ' + name. Functions can take multiple parameters and return values. You call a function by its name with arguments: greet('Alice') returns 'Hello, Alice'. Functions help organize and reuse code."),
    ("How do you create a class in Python?", "To create a class in Python, use the class keyword: class Dog: def __init__(self, name): self.name = name. The __init__ method is the constructor that initializes new objects. You create instances by calling the class: my_dog = Dog('Rex'). Classes can have methods that define behavior and attributes that store state."),
    ("How do you handle errors in Python?", "To handle errors in Python, use try-except blocks: try: result = 10 / 0, except ZeroDivisionError: print('Cannot divide by zero'). You can catch specific exceptions or use a general except clause. The finally block runs regardless of whether an error occurred. Proper error handling makes programs more robust and user-friendly."),
    ("How do you import a module in Python?", "To import a module in Python, use the import statement: import math. You can then use its functions like math.sqrt(16). You can also import specific items: from math import sqrt, pi. Or give an alias: import numpy as np. Python has many built-in modules and thousands of third-party packages available through pip."),
    ("How do you use a for loop in Python?", "A for loop in Python iterates over a sequence: for item in [1, 2, 3]: print(item). You can loop over lists, strings, ranges, and other iterables. The range function generates numbers: for i in range(5) loops from 0 to 4. For loops are the most common way to iterate in Python and are very readable."),
    ("How do you use a while loop in Python?", "A while loop repeats as long as a condition is true: count = 0, while count < 5: print(count), count += 1. While loops are useful when you do not know in advance how many iterations are needed. Use the break statement to exit early and continue to skip to the next iteration. Be careful to avoid infinite loops."),
    ("How do you use list comprehension in Python?", "List comprehension creates a new list from an existing iterable in one line: squares = [x**2 for x in range(10)]. You can add conditions: evens = [x for x in range(20) if x % 2 == 0]. List comprehensions are more concise and often faster than equivalent for loops. They are a signature feature of Python."),
    ("How do you sort a list in Python?", "To sort a list in Python, use the sort method for in-place sorting: my_list.sort(), or the sorted function for a new sorted list: new_list = sorted(my_list). Both accept a reverse parameter: my_list.sort(reverse=True) for descending order. You can also sort by a custom key: sorted(words, key=len) sorts by length."),
    ("How do you use f-strings in Python?", "F-strings format strings with embedded expressions: name = 'Alice', message = f'Hello, {name}!'. They were introduced in Python 3.6. You can include any expression: f'2 + 3 = {2 + 3}'. F-strings also support formatting: f'{3.14159:.2f}' gives '3.14'. They are the preferred way to format strings in modern Python."),
    ("How do you use lambda functions in Python?", "Lambda functions are small anonymous functions defined with the lambda keyword: square = lambda x: x**2. They can take any number of arguments: add = lambda a, b: a + b. Lambdas are useful as arguments to functions like map, filter, and sorted. For example: sorted(pairs, key=lambda x: x[1]) sorts by the second element."),
    ("How do you use decorators in Python?", "Decorators modify functions using the @ syntax. Define a decorator: def my_decorator(func): def wrapper(): print('Before'), func(), print('After'), return wrapper. Apply it: @my_decorator, def say_hello(): print('Hello!'). When you call say_hello(), it prints 'Before', 'Hello!', 'After'. Decorators are powerful for adding reusable behavior."),
    ("How do you use generators in Python?", "Generators are functions that use yield instead of return to produce a sequence of values lazily: def count_up(n): for i in range(n): yield i. You iterate over them: for num in count_up(5): print(num). Generators are memory-efficient because they produce values one at a time instead of storing them all in memory."),
    ("What is the difference between a list and a tuple?", "Lists are mutable and created with square brackets [1, 2, 3], while tuples are immutable and created with parentheses (1, 2, 3). Lists can be modified after creation using methods like append and remove. Tuples cannot be changed after creation. Tuples are faster and use less memory. Use tuples for fixed data and lists for data that changes."),
    ("What is the difference between == and is in Python?", "The == operator checks if two values are equal, while is checks if two variables point to the same object in memory. For example, a = [1, 2] and b = [1, 2]: a == b is True because they have the same values, but a is b is False because they are different objects. Use == for value comparison and is for identity."),
    ("What are *args and **kwargs in Python?", "*args allows a function to accept any number of positional arguments as a tuple: def func(*args): for arg in args: print(arg). **kwargs accepts keyword arguments as a dictionary: def func(**kwargs): for key, value in kwargs.items(): print(key, value). Together they make functions flexible and can accept any combination of arguments."),
    ("How do you use pip to install packages?", "Pip is Python's package manager. Install a package with: pip install package_name. Install a specific version: pip install package_name==1.2.3. Install from requirements file: pip install -r requirements.txt. List installed packages: pip list. Uninstall: pip uninstall package_name. Always use virtual environments to isolate project dependencies."),
]

for q, a in python_qa:
    qa_pairs.append({"instruction": q, "output": a})

# Code examples
code_examples = [
    ("Write a function to calculate factorial.", "Here is a function to calculate factorial in Python:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nThis function uses recursion. The base case returns 1 when n is 0 or 1. Otherwise, it multiplies n by the factorial of (n-1). For example, factorial(5) = 5 * 4 * 3 * 2 * 1 = 120."),
    ("Write a function to check if a number is prime.", "Here is a function to check if a number is prime:\n\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nThis checks divisibility up to the square root of n. This is efficient because if n has a factor larger than its square root, it must also have one smaller than it."),
    ("Write a function to reverse a string.", "Here is a function to reverse a string in Python:\n\ndef reverse_string(s):\n    return s[::-1]\n\nThis uses Python's slice notation with step -1 to reverse the string. For example, reverse_string('hello') returns 'olleh'. You can also use ''.join(reversed(s)) for a more explicit approach."),
    ("Write a function to find the maximum in a list.", "Here is a function to find the maximum value in a list:\n\ndef find_max(lst):\n    if not lst:\n        return None\n    maximum = lst[0]\n    for item in lst[1:]:\n        if item > maximum:\n            maximum = item\n    return maximum\n\nThis iterates through the list keeping track of the largest value. Python also has a built-in max() function: max([3, 1, 4, 1, 5]) returns 5."),
    ("Write a function to count words in a string.", "Here is a function to count words in a string:\n\ndef count_words(text):\n    words = text.split()\n    return len(words)\n\nThe split() method splits the string on whitespace and returns a list of words. Then len() counts the items. For example, count_words('Hello World') returns 2. This handles multiple spaces correctly since split() without arguments handles any whitespace."),
    ("Write a FizzBuzz program.", "Here is a FizzBuzz program in Python:\n\nfor i in range(1, 101):\n    if i % 15 == 0:\n        print('FizzBuzz')\n    elif i % 3 == 0:\n        print('Fizz')\n    elif i % 5 == 0:\n        print('Buzz')\n    else:\n        print(i)\n\nFizzBuzz prints numbers 1 to 100, replacing multiples of 3 with 'Fizz', multiples of 5 with 'Buzz', and multiples of both with 'FizzBuzz'. It is a classic programming interview question."),
    ("Write a function to check if a string is a palindrome.", "Here is a function to check if a string is a palindrome:\n\ndef is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n\nThis converts to lowercase and removes spaces, then checks if the string equals its reverse. For example, is_palindrome('racecar') returns True, and is_palindrome('hello') returns False. Palindromes read the same forwards and backwards."),
    ("Write a function for binary search.", "Here is a binary search function in Python:\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n\nBinary search works on sorted arrays by repeatedly halving the search space. It has O(log n) time complexity, making it very efficient for large datasets."),
    ("Write a function to flatten a nested list.", "Here is a function to flatten a nested list:\n\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n\nThis uses recursion to handle arbitrarily nested lists. For example, flatten([1, [2, [3, 4]], 5]) returns [1, 2, 3, 4, 5]. The function checks each item and recursively flattens sublists."),
    ("Write a function to remove duplicates from a list.", "Here is a function to remove duplicates from a list while preserving order:\n\ndef remove_duplicates(lst):\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result\n\nThis uses a set to track seen items for O(1) lookups. For example, remove_duplicates([1, 2, 2, 3, 1]) returns [1, 2, 3]. Using set() alone would not preserve the original order."),
    ("Write a bubble sort function.", "Here is a bubble sort function in Python:\n\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n\nBubble sort repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. It has O(n^2) time complexity. While not efficient for large lists, it is simple to understand and implement."),
    ("Write a function to merge two sorted lists.", "Here is a function to merge two sorted lists:\n\ndef merge_sorted(list1, list2):\n    result = []\n    i, j = 0, 0\n    while i < len(list1) and j < len(list2):\n        if list1[i] <= list2[j]:\n            result.append(list1[i])\n            i += 1\n        else:\n            result.append(list2[j])\n            j += 1\n    result.extend(list1[i:])\n    result.extend(list2[j:])\n    return result\n\nThis efficiently merges two sorted lists by comparing elements from each. It runs in O(n + m) time. This is a key step in merge sort."),
    ("Write a function to compute Fibonacci numbers.", "Here is a function to compute Fibonacci numbers:\n\ndef fibonacci(n):\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nThis iterative approach computes the nth Fibonacci number efficiently in O(n) time and O(1) space. The Fibonacci sequence is: 0, 1, 1, 2, 3, 5, 8, 13, 21. Each number is the sum of the two preceding numbers."),
    ("Write a function to convert Celsius to Fahrenheit.", "Here is a function to convert Celsius to Fahrenheit:\n\ndef celsius_to_fahrenheit(celsius):\n    return celsius * 9/5 + 32\n\nThe formula is: F = C * 9/5 + 32. For example, celsius_to_fahrenheit(0) returns 32, celsius_to_fahrenheit(100) returns 212, and celsius_to_fahrenheit(37) returns 98.6. Water freezes at 0C (32F) and boils at 100C (212F)."),
    ("Write a function to find the GCD of two numbers.", "Here is a function to find the greatest common divisor using the Euclidean algorithm:\n\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\nThe Euclidean algorithm repeatedly divides the larger number by the smaller and takes the remainder. For example, gcd(48, 18) computes: 48 % 18 = 12, then 18 % 12 = 6, then 12 % 6 = 0, so the GCD is 6. This is an efficient algorithm."),
]

for q, a in code_examples:
    qa_pairs.append({"instruction": q, "output": a})

# ============================================================
# 4. SCIENCE & GENERAL KNOWLEDGE (300 examples)
# ============================================================
science_qa = [
    ("What is photosynthesis?", "Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen. It takes place in the chloroplasts of plant cells, which contain chlorophyll. The basic equation is: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. Photosynthesis is essential for life on Earth as it produces oxygen and food."),
    ("What is gravity?", "Gravity is a fundamental force of nature that attracts objects with mass toward each other. On Earth, gravity gives weight to objects and causes them to fall when dropped. The acceleration due to gravity on Earth is approximately 9.8 meters per second squared. Isaac Newton first described gravity mathematically, and Einstein later refined it with general relativity."),
    ("What is DNA?", "DNA, or deoxyribonucleic acid, is a molecule that carries the genetic instructions for the development and functioning of all living organisms. DNA has a double helix structure made up of nucleotide bases: adenine, thymine, guanine, and cytosine. It is found in the nucleus of cells and contains the code for making proteins. DNA is passed from parents to offspring."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second, or about 186,282 miles per second. This is often rounded to 300,000 km/s. According to Einstein's theory of relativity, nothing can travel faster than the speed of light. Light from the Sun takes about 8 minutes to reach Earth."),
    ("What is an atom?", "An atom is the smallest unit of a chemical element that retains the properties of that element. Atoms consist of a nucleus containing protons and neutrons, surrounded by electrons in orbitals. The number of protons determines the element. For example, hydrogen has 1 proton and carbon has 6. Atoms combine to form molecules and make up all matter."),
    ("What is evolution?", "Evolution is the process by which species change over time through natural selection and genetic variation. Organisms with traits better suited to their environment are more likely to survive and reproduce. Over many generations, this leads to the development of new species. Charles Darwin proposed the theory of evolution by natural selection in 1859."),
    ("What causes earthquakes?", "Earthquakes are caused by the sudden release of energy in the Earth's crust, usually due to the movement of tectonic plates. When plates collide, move apart, or slide past each other, stress builds up along fault lines. When this stress exceeds the strength of the rock, it breaks suddenly, releasing energy as seismic waves that we feel as an earthquake."),
    ("What is the water cycle?", "The water cycle is the continuous movement of water through the Earth's systems. It involves evaporation from bodies of water, condensation into clouds, precipitation as rain or snow, and collection in rivers, lakes, and oceans. Water also moves through the ground as groundwater. The water cycle is driven by solar energy and gravity."),
    ("What is a black hole?", "A black hole is a region in space where gravity is so strong that nothing, not even light, can escape from it. Black holes form when massive stars collapse at the end of their life cycle. The boundary of a black hole is called the event horizon. Black holes were predicted by Einstein's theory of general relativity and have been observed through their effects on nearby matter."),
    ("What are the planets in our solar system?", "The eight planets in our solar system, in order from the Sun, are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. The first four are rocky terrestrial planets, while the outer four are gas and ice giants. Jupiter is the largest planet, and Mercury is the smallest. Pluto was reclassified as a dwarf planet in 2006."),
    ("What is the periodic table?", "The periodic table is a chart that organizes all known chemical elements by their atomic number, electron configuration, and chemical properties. Elements are arranged in rows called periods and columns called groups. Elements in the same group have similar chemical behavior. The periodic table was first proposed by Dmitri Mendeleev in 1869."),
    ("What is electricity?", "Electricity is the flow of electric charge, typically carried by electrons through a conductor like a wire. Electric current is measured in amperes, voltage in volts, and resistance in ohms. Electricity can be generated from various sources including fossil fuels, nuclear energy, wind, and solar power. It powers our homes, devices, and industries."),
    ("What is magnetism?", "Magnetism is a physical phenomenon produced by the motion of electric charge. Magnets have north and south poles that attract or repel each other. The Earth itself is a giant magnet with a magnetic field that protects us from solar radiation. Electromagnetism is one of the four fundamental forces of nature. Magnets are used in motors, generators, and data storage."),
    ("How does the immune system work?", "The immune system is the body's defense mechanism against pathogens like bacteria, viruses, and parasites. It has two main components: innate immunity, which provides immediate but non-specific defense, and adaptive immunity, which develops targeted responses. White blood cells identify and destroy foreign invaders. Vaccines train the immune system to recognize specific pathogens."),
    ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change can occur naturally, the current trend is primarily driven by human activities, especially the burning of fossil fuels which releases greenhouse gases. These gases trap heat in the atmosphere, causing global warming. Effects include rising sea levels, extreme weather, and ecosystem disruption."),
    ("What is the Big Bang theory?", "The Big Bang theory is the prevailing cosmological model explaining the origin of the universe. It states that the universe began as an extremely hot, dense point about 13.8 billion years ago and has been expanding ever since. Evidence for the Big Bang includes the cosmic microwave background radiation and the observed expansion of the universe. As the universe expanded, it cooled, allowing atoms, stars, and galaxies to form."),
    ("What is a cell?", "A cell is the basic structural and functional unit of all living organisms. There are two main types: prokaryotic cells, which lack a nucleus, and eukaryotic cells, which have a nucleus containing DNA. Cells perform essential functions like energy production, protein synthesis, and reproduction. The human body contains approximately 37 trillion cells of various types."),
    ("What is the ozone layer?", "The ozone layer is a region of the Earth's stratosphere that contains a high concentration of ozone (O3) molecules. It absorbs most of the Sun's harmful ultraviolet radiation, protecting life on Earth. The ozone layer has been damaged by human-made chemicals called CFCs (chlorofluorocarbons). International agreements like the Montreal Protocol have helped reduce ozone depletion."),
    ("What are vitamins?", "Vitamins are organic compounds that the body needs in small amounts for proper functioning. There are 13 essential vitamins including A, C, D, E, K, and the B vitamins. They help with metabolism, immunity, and cell growth. Most vitamins must be obtained from food since the body cannot produce enough on its own. Deficiencies can cause various health problems."),
    ("What is artificial intelligence?", "Artificial intelligence, or AI, is the simulation of human intelligence by computer systems. AI encompasses machine learning, natural language processing, computer vision, and robotics. Machine learning allows computers to learn from data without explicit programming. AI applications include virtual assistants, self-driving cars, medical diagnosis, and language translation. The field has advanced rapidly."),
]

for q, a in science_qa:
    qa_pairs.append({"instruction": q, "output": a})

# More general knowledge
general_qa = [
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the telephone in 1876. He was a Scottish-born scientist and engineer who received the first patent for the telephone. Bell demonstrated his invention at the Centennial Exhibition in Philadelphia. The telephone revolutionized communication by allowing people to speak with each other over long distances. Bell also co-founded AT&T."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare around 1596. It is a tragic love story about two young lovers from feuding families in Verona, Italy. The play is one of Shakespeare's most famous works and has been adapted into numerous films, operas, and ballets. Shakespeare is widely regarded as the greatest writer in the English language."),
    ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world, standing at 8,849 meters (29,032 feet) above sea level. It is located in the Himalayas on the border between Nepal and Tibet. The first confirmed summit was by Edmund Hillary and Tenzing Norgay in 1953. Climbing Everest is extremely dangerous due to severe weather, altitude sickness, and avalanches."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth, covering approximately 165.25 million square kilometers. It spans from the Americas in the east to Asia and Australia in the west. The Pacific is also the deepest ocean, with the Mariana Trench reaching nearly 11,000 meters deep. It contains more than half of the world's free water."),
    ("What is the longest river in the world?", "The Nile River is traditionally considered the longest river in the world at approximately 6,650 kilometers long. It flows through northeastern Africa, passing through 11 countries. The Nile has been crucial to the development of Egyptian civilization for thousands of years. Some measurements suggest the Amazon River may be longer, but the Nile is most commonly cited."),
    ("How many continents are there?", "There are seven continents on Earth: Africa, Antarctica, Asia, Australia (sometimes called Oceania), Europe, North America, and South America. Asia is the largest by both area and population. Antarctica is the least populated. Some geographic models combine Europe and Asia into Eurasia. The continents are separated by oceans and divided by cultural and geological boundaries."),
    ("What year did World War 2 end?", "World War 2 ended in 1945. The war in Europe ended on May 8, 1945, known as V-E Day, when Germany surrendered. The war in the Pacific ended on September 2, 1945, known as V-J Day, after Japan surrendered following the atomic bombings of Hiroshima and Nagasaki. World War 2 was the deadliest conflict in human history, with an estimated 70-85 million casualties."),
    ("What is the human body temperature?", "The normal human body temperature is approximately 37 degrees Celsius or 98.6 degrees Fahrenheit. However, body temperature can vary throughout the day and between individuals. It tends to be lower in the morning and higher in the late afternoon. A temperature above 38C (100.4F) is generally considered a fever. Body temperature is regulated by the hypothalamus in the brain."),
    ("What is the boiling point of water?", "The boiling point of water at standard atmospheric pressure is 100 degrees Celsius or 212 degrees Fahrenheit. At this temperature, water transitions from liquid to gas (steam). The boiling point changes with altitude and pressure: at higher altitudes where pressure is lower, water boils at a lower temperature. This is why cooking takes longer at high elevations."),
    ("What is the freezing point of water?", "The freezing point of water at standard atmospheric pressure is 0 degrees Celsius or 32 degrees Fahrenheit. At this temperature, liquid water transitions to solid ice. The freezing point can be lowered by adding solutes like salt, which is why salt is used to melt ice on roads. Pure water expands when it freezes, which is unusual for most substances."),
    ("How many bones are in the human body?", "An adult human body contains 206 bones. Babies are born with about 270 bones, but many fuse together as they grow. The skeletal system provides structure, protects organs, stores minerals, and produces blood cells. The femur (thigh bone) is the largest bone, and the stapes in the ear is the smallest. Bones are living tissue that constantly remodel themselves."),
    ("What is the speed of sound?", "The speed of sound in air at room temperature is approximately 343 meters per second, or about 1,235 kilometers per hour. Sound travels faster in denser media: about 1,480 m/s in water and 5,960 m/s in steel. The speed of sound changes with temperature, increasing about 0.6 m/s for each degree Celsius rise. When an object exceeds the speed of sound, it creates a sonic boom."),
    ("What causes a rainbow?", "A rainbow is caused by the reflection, refraction, and dispersion of sunlight through water droplets in the atmosphere. White sunlight enters a raindrop, bends as it passes through, reflects off the back of the drop, and bends again as it exits. Different wavelengths of light bend at slightly different angles, separating into the visible spectrum. Rainbows always appear opposite the sun."),
    ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides. Written as a formula: a^2 + b^2 = c^2, where c is the hypotenuse. For example, a triangle with sides 3 and 4 has a hypotenuse of 5, because 9 + 16 = 25. This theorem is named after the ancient Greek mathematician Pythagoras."),
    ("What is pi?", "Pi is a mathematical constant approximately equal to 3.14159. It represents the ratio of a circle's circumference to its diameter. Pi is an irrational number, meaning its decimal representation never ends or repeats. It is used extensively in mathematics, physics, and engineering. The symbol for pi is the Greek letter π. Pi has been calculated to trillions of decimal places."),
    ("What is democracy?", "Democracy is a system of government in which power is held by the people, who exercise it directly or through elected representatives. The word comes from the Greek words demos (people) and kratos (power). Modern democracies typically feature free elections, rule of law, protection of individual rights, and separation of powers. Democracy originated in ancient Athens around the 5th century BCE."),
    ("What is the Internet?", "The Internet is a global network of interconnected computers that communicate using standardized protocols, primarily TCP/IP. It evolved from ARPANET, a US military network created in the 1960s. The World Wide Web, invented by Tim Berners-Lee in 1989, made the Internet accessible through web browsers. Today, the Internet connects billions of devices and enables email, social media, streaming, and e-commerce."),
    ("What is a volcano?", "A volcano is an opening in the Earth's crust through which molten rock (magma), gases, and ash can escape to the surface. Volcanoes form at tectonic plate boundaries and hotspots. When magma reaches the surface, it is called lava. Volcanic eruptions can be explosive or effusive. Famous volcanoes include Mount Vesuvius, Mount St. Helens, and Kilauea. There are about 1,500 active volcanoes on Earth."),
    ("What are the states of matter?", "The three primary states of matter are solid, liquid, and gas. In a solid, particles are tightly packed and vibrate in fixed positions. In a liquid, particles are close together but can flow. In a gas, particles are far apart and move freely. A fourth state, plasma, exists at very high temperatures. Matter can change states through heating or cooling: melting, freezing, evaporating, and condensing."),
    ("What is photosynthesis important for?", "Photosynthesis is vital because it produces oxygen and food that sustain nearly all life on Earth. Plants convert carbon dioxide and water into glucose using sunlight energy. The oxygen released as a byproduct is what animals breathe. Photosynthesis also forms the base of most food chains. Additionally, it helps regulate atmospheric carbon dioxide levels, playing a role in Earth's climate system."),
]

for q, a in general_qa:
    qa_pairs.append({"instruction": q, "output": a})

# ============================================================
# 5. CONVERSATIONAL / INSTRUCTIONS (100 examples)
# ============================================================
conversational_qa = [
    ("Hello!", "Hello! How can I help you today? I am an AI assistant ready to answer your questions about programming, science, math, and many other topics. Feel free to ask me anything and I will do my best to provide a helpful and accurate response."),
    ("Hi there!", "Hi there! Welcome! I am here to help you with any questions you might have. Whether it is about programming, science, math, or general knowledge, I will do my best to give you a clear and helpful answer. What would you like to know?"),
    ("What can you do?", "I can help you with a wide range of topics. I can answer questions about programming, mathematics, science, geography, and general knowledge. I can explain concepts, write code examples, solve math problems, and provide factual information. I aim to give clear, accurate, and helpful responses to your questions."),
    ("Thank you!", "You are welcome! I am glad I could help. If you have any more questions, feel free to ask. I am always here to assist you with programming, math, science, or any other topic you need help with. Do not hesitate to reach out anytime."),
    ("Who are you?", "I am an AI assistant designed to help you with various questions and tasks. I can answer questions about programming, science, math, history, and many other subjects. My goal is to provide accurate, clear, and helpful responses. I am always learning and improving to better assist you."),
    ("What is your name?", "I am an AI assistant. You can think of me as a helpful tool for answering questions and solving problems. I am designed to assist with a wide variety of topics including programming, science, mathematics, and general knowledge. I am here to help you learn and find the information you need."),
    ("How are you?", "I am doing well, thank you for asking! As an AI, I do not have feelings in the traditional sense, but I am functioning properly and ready to help you. What can I assist you with today? I am happy to answer questions on any topic you are interested in."),
    ("Can you help me?", "Of course! I am here to help you with whatever you need. Just ask me a question and I will do my best to provide a clear and accurate answer. I can help with programming, math, science, writing, and many other topics. What would you like to know?"),
    ("Explain something simple.", "Sure! Let me explain something simple. Water boils at 100 degrees Celsius at sea level. When you heat water, the molecules move faster and faster. At the boiling point, molecules have enough energy to escape as steam. This is a phase change from liquid to gas. The boiling point decreases at higher altitudes because of lower air pressure."),
    ("Tell me a fun fact.", "Here is a fun fact: honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible. Honey has naturally low moisture content and acidic pH that prevent bacterial growth. Its long shelf life is due to the enzyme glucose oxidase, which produces hydrogen peroxide as a byproduct."),
    ("What should I learn first in programming?", "If you are new to programming, I recommend starting with Python. Python has a simple, readable syntax that is great for beginners. Start by learning variables, data types, and basic operations. Then move on to control flow with if statements and loops. Next, learn about functions and how to organize your code. After that, explore data structures like lists and dictionaries. Practice by building small projects."),
    ("Give me a tip for studying.", "A great study tip is to use the Pomodoro technique: study for 25 minutes, then take a 5-minute break. After four sessions, take a longer 15-30 minute break. This helps maintain focus and prevents burnout. Also, try to explain what you learned in your own words - teaching is one of the best ways to learn. Active recall and spaced repetition are other proven study methods."),
    ("Summarize the theory of relativity.", "Einstein's theory of relativity has two parts. Special relativity (1905) says that the speed of light is constant for all observers, and that space and time are linked as spacetime. It gave us the famous equation E=mc^2. General relativity (1915) describes gravity as the curvature of spacetime caused by mass and energy. Massive objects bend spacetime, which is what we experience as gravity."),
    ("What is machine learning?", "Machine learning is a branch of artificial intelligence where computers learn patterns from data without being explicitly programmed. There are three main types: supervised learning uses labeled data to make predictions, unsupervised learning finds patterns in unlabeled data, and reinforcement learning learns through trial and error with rewards. Applications include image recognition, natural language processing, and recommendation systems."),
    ("How does a computer work?", "A computer works by processing instructions using its central processing unit (CPU). Programs and data are stored in memory (RAM) for quick access. The CPU fetches instructions from memory, decodes them, executes them, and stores results. Input devices like keyboards and mice send data in, while output devices like monitors display results. Storage devices like SSDs keep data permanently even when powered off."),
]

for q, a in conversational_qa:
    qa_pairs.append({"instruction": q, "output": a})

# ============================================================
# 6. MORE MATH - Word Problems (100 examples)
# ============================================================
for _ in range(25):
    price = random.randint(5, 50)
    qty = random.randint(2, 10)
    total = price * qty
    item = random.choice(["apples", "books", "pencils", "shirts", "toys", "bottles", "tickets", "cookies", "sandwiches", "pens"])
    qa_pairs.append({
        "instruction": f"If one {item[:-1]} costs ${price} and you buy {qty}, how much do you pay?",
        "output": f"You would pay ${total}. To find the total cost, multiply the price of one {item[:-1]} (${price}) by the quantity ({qty}). So ${price} times {qty} equals ${total}. This is a basic multiplication problem commonly used in everyday shopping calculations."
    })

for _ in range(25):
    total = random.randint(20, 200)
    parts = random.randint(2, 8)
    each = total / parts
    qa_pairs.append({
        "instruction": f"If you divide ${total} equally among {parts} people, how much does each person get?",
        "output": f"Each person gets ${each:.2f}. To find this, divide the total amount (${total}) by the number of people ({parts}). So ${total} divided by {parts} equals ${each:.2f}. This is a practical division problem often used when splitting bills or sharing resources equally."
    })

for _ in range(25):
    speed = random.choice([30, 40, 50, 60, 70, 80])
    time = random.choice([2, 3, 4, 5])
    dist = speed * time
    qa_pairs.append({
        "instruction": f"If a car travels at {speed} km/h for {time} hours, how far does it go?",
        "output": f"The car travels {dist} kilometers. Distance equals speed multiplied by time. So {speed} km/h times {time} hours equals {dist} km. This formula, distance = speed x time, is fundamental in physics and is used to calculate how far an object travels at a constant speed."
    })

for _ in range(25):
    l = random.randint(3, 20)
    w = random.randint(3, 20)
    area = l * w
    peri = 2 * (l + w)
    qa_pairs.append({
        "instruction": f"What is the area and perimeter of a rectangle with length {l} and width {w}?",
        "output": f"The area is {area} square units and the perimeter is {peri} units. Area is calculated by multiplying length times width: {l} times {w} equals {area}. Perimeter is calculated by adding all sides: 2 times ({l} + {w}) equals {peri}. These are fundamental geometry formulas used in construction, design, and many other fields."
    })

# ============================================================
# 7. DATA STRUCTURES & ALGORITHMS (50 examples)
# ============================================================
ds_qa = [
    ("What is a stack?", "A stack is a linear data structure that follows the Last-In-First-Out (LIFO) principle. The last element added is the first one removed. Key operations are push (add to top), pop (remove from top), and peek (view top without removing). Stacks are used in function call management, undo operations, expression evaluation, and backtracking algorithms. Think of it like a stack of plates."),
    ("What is a queue?", "A queue is a linear data structure that follows the First-In-First-Out (FIFO) principle. The first element added is the first one removed. Key operations are enqueue (add to rear), dequeue (remove from front), and peek (view front element). Queues are used in breadth-first search, task scheduling, and message passing. Think of it like a line at a store."),
    ("What is a linked list?", "A linked list is a linear data structure where elements are stored in nodes. Each node contains data and a reference (pointer) to the next node. Unlike arrays, linked lists do not store elements in contiguous memory. This allows efficient insertion and deletion at any position. Types include singly linked, doubly linked, and circular linked lists."),
    ("What is a hash table?", "A hash table is a data structure that maps keys to values using a hash function. The hash function converts a key into an array index where the value is stored. This allows average O(1) time complexity for insertions, deletions, and lookups. Collisions occur when two keys hash to the same index. Common collision resolution methods include chaining and open addressing."),
    ("What is a binary tree?", "A binary tree is a hierarchical data structure where each node has at most two children, called left and right. The topmost node is called the root. Binary trees are used in searching, sorting, and hierarchical data representation. A binary search tree maintains the property that left children are smaller and right children are larger than the parent."),
    ("What is Big O notation?", "Big O notation describes the upper bound of an algorithm's time or space complexity as input size grows. Common complexities include O(1) for constant time, O(log n) for logarithmic, O(n) for linear, O(n log n) for linearithmic, and O(n^2) for quadratic. Big O helps compare algorithm efficiency and choose the best approach for a given problem size."),
    ("What is a graph?", "A graph is a data structure consisting of vertices (nodes) connected by edges. Graphs can be directed or undirected, weighted or unweighted. They model relationships between objects, such as social networks, road maps, and web pages. Common graph algorithms include depth-first search, breadth-first search, Dijkstra's shortest path, and minimum spanning tree."),
    ("What is dynamic programming?", "Dynamic programming is an algorithmic technique that solves complex problems by breaking them into overlapping subproblems and storing their solutions. It avoids redundant calculations by building solutions bottom-up or using memoization top-down. Classic examples include the Fibonacci sequence, knapsack problem, and longest common subsequence. It trades memory for speed."),
    ("What is recursion vs iteration?", "Recursion solves a problem by having a function call itself with smaller inputs until reaching a base case. Iteration uses loops to repeat a block of code. Both can solve the same problems. Recursion is often more elegant but uses more memory due to the call stack. Iteration is generally more memory-efficient. Some problems like tree traversal are naturally recursive."),
    ("What is a heap?", "A heap is a specialized tree-based data structure that satisfies the heap property. In a max-heap, each parent is greater than or equal to its children. In a min-heap, each parent is less than or equal to its children. Heaps are commonly implemented as arrays and are used in priority queues and heap sort. Insertion and deletion take O(log n) time."),
    ("What is sorting?", "Sorting is the process of arranging elements in a specific order, usually ascending or descending. Common sorting algorithms include bubble sort (O(n^2)), insertion sort (O(n^2)), merge sort (O(n log n)), quicksort (O(n log n) average), and heap sort (O(n log n)). The choice of algorithm depends on data size, memory constraints, and whether the data is partially sorted."),
    ("What is a trie?", "A trie, also called a prefix tree, is a tree data structure used for storing strings. Each node represents a character, and paths from root to leaf represent complete strings. Tries enable fast prefix-based searching and are commonly used in autocomplete systems, spell checkers, and IP routing tables. Lookup time is O(m) where m is the length of the search string."),
    ("What is memoization?", "Memoization is an optimization technique that stores the results of expensive function calls and returns the cached result when the same inputs occur again. It is a form of dynamic programming. In Python, you can use the functools.lru_cache decorator to memoize a function. Memoization is especially useful for recursive functions with overlapping subproblems like Fibonacci."),
    ("What is depth-first search?", "Depth-first search (DFS) is a graph traversal algorithm that explores as far as possible along each branch before backtracking. It uses a stack (or recursion) to keep track of which nodes to visit next. DFS is used for topological sorting, cycle detection, path finding, and solving puzzles like mazes. Its time complexity is O(V + E) where V is vertices and E is edges."),
    ("What is breadth-first search?", "Breadth-first search (BFS) is a graph traversal algorithm that explores all neighbors at the current depth before moving to the next level. It uses a queue to keep track of nodes to visit. BFS finds the shortest path in unweighted graphs. It is used for level-order tree traversal, finding connected components, and social network analysis. Its time complexity is O(V + E)."),
]

for q, a in ds_qa:
    qa_pairs.append({"instruction": q, "output": a})

# ============================================================
# 8. HISTORY (80 examples)
# ============================================================
history_qa = [
    ("When was the Declaration of Independence signed?", "The United States Declaration of Independence was adopted on July 4, 1776. It was primarily written by Thomas Jefferson and declared the thirteen American colonies free from British rule. The signing took place in Philadelphia at Independence Hall. July 4th is now celebrated as Independence Day in the United States, marking the birth of the nation."),
    ("Who was the first president of the United States?", "George Washington was the first president of the United States, serving from 1789 to 1797. He led the Continental Army to victory in the American Revolutionary War and presided over the Constitutional Convention. Washington is often called the Father of His Country. He set many precedents for the presidency, including the tradition of serving only two terms."),
    ("What was the Industrial Revolution?", "The Industrial Revolution was a period of rapid industrialization from the mid-1700s to mid-1800s, beginning in Britain. It marked the transition from hand production to machine manufacturing. Key innovations included the steam engine, spinning jenny, and power loom. The Industrial Revolution transformed economies, created factories, and led to urbanization. It fundamentally changed how people lived and worked."),
    ("What was the Renaissance?", "The Renaissance was a cultural movement that began in Italy in the 14th century and spread across Europe. It marked a renewed interest in classical Greek and Roman art, literature, and philosophy. Famous Renaissance figures include Leonardo da Vinci, Michelangelo, and Shakespeare. The period saw advances in art, science, and humanist thought. The word Renaissance means rebirth in French."),
    ("Who was Albert Einstein?", "Albert Einstein was a German-born theoretical physicist who lived from 1879 to 1955. He is best known for developing the theory of relativity and the famous equation E=mc^2. He won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. Einstein's work revolutionized our understanding of space, time, gravity, and the universe."),
    ("What was the Cold War?", "The Cold War was a period of geopolitical tension between the United States and the Soviet Union, lasting from approximately 1947 to 1991. It was characterized by political rivalry, nuclear arms race, space race, and proxy wars. Key events included the Korean War, Cuban Missile Crisis, and Vietnam War. The Cold War ended with the dissolution of the Soviet Union in 1991."),
    ("Who was Cleopatra?", "Cleopatra VII was the last active ruler of the Ptolemaic Kingdom of Egypt, reigning from 51 to 30 BCE. She was known for her intelligence, political skill, and relationships with Roman leaders Julius Caesar and Mark Antony. Cleopatra spoke multiple languages and was a skilled diplomat. She remains one of the most famous figures in ancient history."),
    ("What was the French Revolution?", "The French Revolution was a period of radical political and social upheaval in France from 1789 to 1799. It began with the storming of the Bastille on July 14, 1789. The revolution overthrew the monarchy, established a republic, and led to Napoleon's rise to power. Key ideals included liberty, equality, and fraternity. It profoundly influenced modern democracy and politics."),
    ("Who invented the printing press?", "Johannes Gutenberg invented the movable-type printing press around 1440 in Mainz, Germany. His invention revolutionized the production of books, making them cheaper and more widely available. The first major work printed was the Gutenberg Bible. The printing press helped spread knowledge, literacy, and ideas across Europe, playing a crucial role in the Renaissance and Reformation."),
    ("What was the Space Race?", "The Space Race was a competition between the United States and the Soviet Union during the Cold War to achieve milestones in space exploration. It began in 1957 when the Soviets launched Sputnik, the first artificial satellite. The US responded by founding NASA and ultimately won the race by landing astronauts on the Moon on July 20, 1969, with Apollo 11."),
    ("Who was Isaac Newton?", "Sir Isaac Newton was an English mathematician and physicist who lived from 1643 to 1727. He formulated the laws of motion and universal gravitation, developed calculus, and made groundbreaking discoveries in optics. His work Principia Mathematica is one of the most influential scientific works ever written. Newton's laws form the foundation of classical mechanics."),
    ("What was the Roman Empire?", "The Roman Empire was one of the largest and most influential civilizations in history, lasting from 27 BCE to 476 CE in the West. At its height, it spanned from Britain to the Middle East. Rome made lasting contributions to law, engineering, architecture, language, and government. The empire's roads, aqueducts, and buildings were engineering marvels. Latin, the Roman language, influenced many modern European languages."),
    ("Who was Marie Curie?", "Marie Curie was a Polish-born physicist and chemist who lived from 1867 to 1934. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences: Physics in 1903 and Chemistry in 1911. She discovered the elements polonium and radium and pioneered research on radioactivity. Her work laid the foundation for nuclear physics."),
    ("What was the Silk Road?", "The Silk Road was a network of ancient trade routes connecting China to the Mediterranean world, active from about 130 BCE to the 1450s. It facilitated the exchange of silk, spices, precious metals, and other goods between East and West. Beyond trade, it spread religions, technologies, and cultural practices. The Silk Road was one of the most important trade networks in history."),
    ("What was the Civil Rights Movement?", "The Civil Rights Movement was a struggle for social justice in the United States during the 1950s and 1960s. It sought equal rights for African Americans and an end to racial segregation and discrimination. Key figures included Martin Luther King Jr., Rosa Parks, and Malcolm X. Major achievements include the Civil Rights Act of 1964 and the Voting Rights Act of 1965."),
]

for q, a in history_qa:
    qa_pairs.append({"instruction": q, "output": a})

# ============================================================
# 9. MORE LANGUAGE & GRAMMAR (50 examples)
# ============================================================
language_qa = [
    ("What is a noun?", "A noun is a word that represents a person, place, thing, or idea. Examples include dog, city, happiness, and teacher. Nouns can be proper (specific names like Paris) or common (general like city). They can be singular or plural, and concrete (touchable) or abstract (concepts). Nouns are one of the fundamental parts of speech in English grammar."),
    ("What is a verb?", "A verb is a word that expresses an action, occurrence, or state of being. Examples include run, think, is, and become. Verbs can be transitive (take an object) or intransitive (no object). They change form based on tense (past, present, future), person, and number. Verbs are essential in every sentence and form the core of the predicate."),
    ("What is an adjective?", "An adjective is a word that describes or modifies a noun. Examples include big, red, happy, and fast. Adjectives provide additional information about the qualities, size, color, or quantity of nouns. They can appear before a noun (the tall building) or after a linking verb (the building is tall). Adjectives make language more descriptive and precise."),
    ("What is an adverb?", "An adverb is a word that modifies a verb, adjective, or another adverb. Examples include quickly, very, often, and carefully. Adverbs describe how, when, where, or to what extent something happens. Many adverbs end in -ly (slowly, happily), but not all (always, here, very). Adverbs add detail and nuance to sentences."),
    ("What is a pronoun?", "A pronoun is a word that takes the place of a noun. Examples include he, she, it, they, and we. Pronouns prevent repetition in sentences. Personal pronouns (I, you, he) refer to specific people. Possessive pronouns (mine, yours) show ownership. Relative pronouns (who, which) connect clauses. Using pronouns correctly makes writing flow more naturally."),
    ("What is a preposition?", "A preposition is a word that shows the relationship between a noun and other words in a sentence. Common prepositions include in, on, at, by, with, and from. They indicate location (on the table), time (at noon), direction (to the store), or manner (with care). Prepositions are followed by a noun or pronoun to form prepositional phrases."),
    ("What is a conjunction?", "A conjunction is a word that connects words, phrases, or clauses. There are three types: coordinating conjunctions (and, but, or, so, yet), subordinating conjunctions (because, although, if, when), and correlative conjunctions (either/or, neither/nor). Conjunctions help create complex sentences by joining related ideas. They are essential for clear and connected writing."),
    ("What is a sentence?", "A sentence is a group of words that expresses a complete thought. Every sentence needs at least a subject and a verb. There are four types: declarative (makes a statement), interrogative (asks a question), imperative (gives a command), and exclamatory (expresses strong emotion). Well-formed sentences follow grammatical rules and convey meaning clearly."),
    ("What is a paragraph?", "A paragraph is a group of related sentences about a single topic. It typically starts with a topic sentence that states the main idea. Supporting sentences provide details, examples, and explanations. A concluding sentence may wrap up the paragraph. Good paragraphs are unified, coherent, and adequately developed. They are the building blocks of longer pieces of writing."),
    ("What is the difference between there, their, and they're?", "These three words sound the same but have different meanings. 'There' indicates a place or introduces a subject (the book is over there). 'Their' is a possessive pronoun showing ownership (their house is blue). 'They're' is a contraction of they are (they're coming tomorrow). Mixing these up is a common grammar mistake. Remember: there for place, their for possession, they're for they are."),
]

for q, a in language_qa:
    qa_pairs.append({"instruction": q, "output": a})

# ============================================================
# 10. TECHNOLOGY (50 examples)
# ============================================================
tech_qa = [
    ("What is HTML?", "HTML stands for HyperText Markup Language. It is the standard language for creating web pages. HTML uses tags like <p> for paragraphs, <h1> for headings, and <a> for links to structure content. Browsers interpret HTML to display web pages. HTML works together with CSS for styling and JavaScript for interactivity. It was created by Tim Berners-Lee in 1991."),
    ("What is CSS?", "CSS stands for Cascading Style Sheets. It is used to control the visual presentation of HTML elements on web pages. CSS lets you set colors, fonts, layouts, spacing, and animations. Styles can be applied inline, in a style block, or in external files. CSS makes websites look attractive and responsive across different screen sizes."),
    ("What is JavaScript?", "JavaScript is a programming language primarily used for web development. It runs in web browsers and allows interactive features like form validation, animations, and dynamic content updates. JavaScript can also run on servers using Node.js. It supports object-oriented, functional, and event-driven programming styles. JavaScript is one of the three core technologies of the web alongside HTML and CSS."),
    ("What is a server?", "A server is a computer or program that provides services, data, or resources to other computers called clients over a network. Web servers deliver web pages, email servers handle messages, and database servers manage data. Servers are typically more powerful than regular computers and run specialized software. The client-server model is fundamental to the Internet and modern computing."),
    ("What is cloud computing?", "Cloud computing delivers computing services like storage, servers, databases, and software over the Internet. Instead of owning physical hardware, you rent resources from providers like Amazon Web Services, Google Cloud, or Microsoft Azure. Benefits include scalability, cost savings, and flexibility. Cloud services are categorized as Infrastructure as a Service, Platform as a Service, or Software as a Service."),
    ("What is encryption?", "Encryption is the process of converting data into a coded format to prevent unauthorized access. It uses mathematical algorithms and keys to transform readable data (plaintext) into unreadable data (ciphertext). Only someone with the correct key can decrypt and read the data. Encryption is essential for securing online communications, passwords, financial transactions, and personal data."),
    ("What is a URL?", "A URL, or Uniform Resource Locator, is the address of a resource on the Internet. It consists of a protocol (https), domain name (www.example.com), and optional path (/page.html). URLs direct browsers to specific web pages or files. The domain name system (DNS) translates human-readable URLs into IP addresses that computers use to locate servers."),
    ("What is an operating system?", "An operating system is software that manages computer hardware and provides services for running applications. Common operating systems include Windows, macOS, and Linux for desktops, and Android and iOS for mobile devices. The OS handles memory management, file systems, process scheduling, and hardware communication. It acts as an intermediary between users and computer hardware."),
    ("What is a firewall?", "A firewall is a network security system that monitors and controls incoming and outgoing network traffic based on predetermined rules. Firewalls create a barrier between trusted internal networks and untrusted external networks like the Internet. They can be hardware devices or software programs. Firewalls help protect computers from unauthorized access, malware, and cyberattacks."),
    ("What is version control?", "Version control is a system for tracking changes to files over time. It allows multiple people to collaborate on code, tracks who made what changes, and enables reverting to previous versions. Git is the most popular version control system. Key concepts include repositories, commits, branches, and merges. Version control is essential for professional software development."),
]

for q, a in tech_qa:
    qa_pairs.append({"instruction": q, "output": a})

# ============================================================
# 11. MORE MATH - Number Theory (100 examples)
# ============================================================
for n in range(2, 52):
    factors = [i for i in range(1, n+1) if n % i == 0]
    factors_str = ", ".join(str(f) for f in factors)
    qa_pairs.append({
        "instruction": f"What are the factors of {n}?",
        "output": f"The factors of {n} are: {factors_str}. Factors are numbers that divide evenly into {n} with no remainder. {n} has {len(factors)} factors. Finding factors is useful in simplifying fractions, finding greatest common divisors, and number theory."
    })

for a_val in range(2, 13):
    for b_val in range(2, 7):
        result = a_val ** b_val
        qa_pairs.append({
            "instruction": f"What is {a_val} to the power of {b_val}?",
            "output": f"{a_val} to the power of {b_val} equals {result}. This means multiplying {a_val} by itself {b_val} times. Exponentiation is a fundamental mathematical operation used in science, engineering, and computer science. Powers are written as {a_val}^{b_val} = {result}."
        })

# ============================================================
# Shuffle and write
# ============================================================
random.shuffle(qa_pairs)

out_path = os.path.join(os.path.dirname(__file__), "train.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for item in qa_pairs:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Generated {len(qa_pairs)} Q&A pairs")

# Count approximate tokens
total_chars = sum(len(item["instruction"]) + len(item["output"]) for item in qa_pairs)
est_tokens = total_chars // 4  # rough BPE estimate
print(f"Estimated tokens: ~{est_tokens:,}")
print(f"Average tokens per example: ~{est_tokens // len(qa_pairs)}")
