import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tkinter as tk
from tkinter import scrolledtext

nltk.download('stopwords')

data = {
    "question": [
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is a string in C?",
        "How to use strings in C?",
        "What is a structure in C?",
        "How to define a structure in C?",
        "What is dynamic memory allocation in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "How to use malloc in C?",
        "What is the difference between malloc and calloc?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "How to declare a variable in C?",
        "What are the data types in C?",
        "How to initialize a variable in C?",
        "What is the scope of a variable in C?",
        "What is a global variable in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is a local variable in C?",
        "What is a pointer in C?",
        "How to use pointers in C?",
        "What is a function in C?",
        "How to define a function in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is a loop in C?",
        "How to use a for loop in C?",
        "How to use a while loop in C?",
        "What is an array in C?",
        "How to declare an array in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is a string in C?",
        "How to use strings in C?",
        "What is a structure in C?",
        "How to define a structure in C?",
        "What is dynamic memory allocation in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "How to use malloc in C?",
        "What is the difference between malloc and calloc?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is a string in C?",
        "How to use strings in C?",
        "What is a structure in C?",
        "How to define a structure in C?",
        "What is dynamic memory allocation in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is a string in C?",
        "How to use strings in C?",
        "What is a structure in C?",
        "How to define a structure in C?",
        "What is dynamic memory allocation in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "How to use malloc in C?",
        "What is the difference between malloc and calloc?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "How to declare a variable in C?",
        "What are the data types in C?",
        "How to initialize a variable in C?",
        "What is the scope of a variable in C?",
        "What is a global variable in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is a local variable in C?",
        "What is a pointer in C?",
        "How to use pointers in C?",
        "What is a function in C?",
        "How to define a function in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is a loop in C?",
        "How to use a for loop in C?",
        "How to use a while loop in C?",
        "What is an array in C?",
        "How to declare an array in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is a string in C?",
        "How to use strings in C?",
        "What is a structure in C?",
        "How to define a structure in C?",
        "What is dynamic memory allocation in C?",
        "What is the header file library in C?",
        "What happens with blank lines?",
        "Why is return 0 used?",
        "How to print an output?",
        "What will happen if a semicolon is missed?",
        "Why use double quotation marks?",
        "How to add a new line?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "How to use malloc in C?",
        "What is the difference between malloc and calloc?",
        "How to create a horizontal tab?",
        "What is a comment in C and how to add it?",
        "What is a variable in C?",
        "What is a string in C?",
        "How to use strings in C?",
        "What is a structure in C?",
        "How to define a structure in C?",
        "What is dynamic memory allocation in C?",
        "What is the header file library in C?",
        "What happens with blank lines?"
    ],
    "answer": [
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "A string is an array of characters ending with a null character (\\0).",
        "You can use strings by declaring them as character arrays or using string functions like `strcpy` and `strcat`.",
        "A structure is a user-defined data type that groups related variables.",
        "You can define a structure using the syntax: `struct structure_name { ... };`.",
        "Dynamic memory allocation allows you to allocate memory at runtime.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "You can use `malloc` to allocate memory: `ptr = (cast_type*) malloc(size);`.",
        "`malloc` allocates memory without initializing it, while `calloc` allocates and initializes memory to zero.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "You can declare a variable using the syntax: `data_type variable_name;`.",
        "Common data types in C include int, float, char, and double.",
        "You can initialize a variable at the time of declaration: `data_type variable_name = value;`.",
        "The scope of a variable determines where it can be accessed in the program.",
        "A global variable is declared outside all functions and can be accessed anywhere in the program.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "A local variable is declared inside a function and can only be accessed within that function.",
        "A pointer is a variable that stores the memory address of another variable.",
        "You can use pointers by declaring them with the `*` symbol and accessing the address with the `&` symbol.",
        "A function is a block of code that performs a specific task.",
        "You can define a function using the syntax: `return_type function_name(parameters) { ... }`.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "A loop is used to execute a block of code repeatedly.",
        "You can use a for loop with the syntax: `for (initialization; condition; increment) { ... }`.",
        "You can use a while loop with the syntax: `while (condition) { ... }`.",
        "An array is a collection of elements of the same data type.",
        "You can declare an array using the syntax: `data_type array_name[size];`.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "A string is an array of characters ending with a null character (\\0).",
        "You can use strings by declaring them as character arrays or using string functions like `strcpy` and `strcat`.",
        "A structure is a user-defined data type that groups related variables.",
        "You can define a structure using the syntax: `struct structure_name { ... };`.",
        "Dynamic memory allocation allows you to allocate memory at runtime.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "You can use `malloc` to allocate memory: `ptr = (cast_type*) malloc(size);`.",
        "`malloc` allocates memory without initializing it, while `calloc` allocates and initializes memory to zero.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "A string is an array of characters ending with a null character (\\0).",
        "You can use strings by declaring them as character arrays or using string functions like `strcpy` and `strcat`.",
        "A structure is a user-defined data type that groups related variables.",
        "You can define a structure using the syntax: `struct structure_name { ... };`.",
        "Dynamic memory allocation allows you to allocate memory at runtime.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "A string is an array of characters ending with a null character (\\0).",
        "You can use strings by declaring them as character arrays or using string functions like `strcpy` and `strcat`.",
        "A structure is a user-defined data type that groups related variables.",
        "You can define a structure using the syntax: `struct structure_name { ... };`.",
        "Dynamic memory allocation allows you to allocate memory at runtime.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "You can use `malloc` to allocate memory: `ptr = (cast_type*) malloc(size);`.",
        "`malloc` allocates memory without initializing it, while `calloc` allocates and initializes memory to zero.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "You can declare a variable using the syntax: `data_type variable_name;`.",
        "Common data types in C include int, float, char, and double.",
        "You can initialize a variable at the time of declaration: `data_type variable_name = value;`.",
        "The scope of a variable determines where it can be accessed in the program.",
        "A global variable is declared outside all functions and can be accessed anywhere in the program.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "A local variable is declared inside a function and can only be accessed within that function.",
        "A pointer is a variable that stores the memory address of another variable.",
        "You can use pointers by declaring them with the `*` symbol and accessing the address with the `&` symbol.",
        "A function is a block of code that performs a specific task.",
        "You can define a function using the syntax: `return_type function_name(parameters) { ... }`.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "A loop is used to execute a block of code repeatedly.",
        "You can use a for loop with the syntax: `for (initialization; condition; increment) { ... }`.",
        "You can use a while loop with the syntax: `while (condition) { ... }`.",
        "An array is a collection of elements of the same data type.",
        "You can declare an array using the syntax: `data_type array_name[size];`.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "A string is an array of characters ending with a null character (\\0).",
        "You can use strings by declaring them as character arrays or using string functions like `strcpy` and `strcat`.",
        "A structure is a user-defined data type that groups related variables.",
        "You can define a structure using the syntax: `struct structure_name { ... };`.",
        "Dynamic memory allocation allows you to allocate memory at runtime.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable.",
        "return 0 ends the main() function.",
        "printf() function",
        "If you forget the semicolon (;), an error will occur and the program will not run.",
        "If you forget the double quotes, an error occurs.",
        "You can use the \\n character.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "You can use `malloc` to allocate memory: `ptr = (cast_type*) malloc(size);`.",
        "`malloc` allocates memory without initializing it, while `calloc` allocates and initializes memory to zero.",
        "You can use the \\t character.",
        "Comments can be used to explain code and to make it more readable. They can also be used to prevent execution when testing alternative code. Comments can be single-lined (//) or multi-lined (/* and */).",
        "A variable is a container for storing data values in C.",
        "A string is an array of characters ending with a null character (\\0).",
        "You can use strings by declaring them as character arrays or using string functions like `strcpy` and `strcat`.",
        "A structure is a user-defined data type that groups related variables.",
        "You can define a structure using the syntax: `struct structure_name { ... };`.",
        "Dynamic memory allocation allows you to allocate memory at runtime.",
        "#include<stdio.h>",
        "C ignores white space. But we use it to make the code more readable."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    if text:
        # Lowercase, remove stopwords, and apply stemming
        return " ".join([stemmer.stem(word) for word in text.lower().split() if word not in stop_words])
    return ""

df['question'] = df['question'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(df['question'], df['answer'], test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=0.1))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

class ProgrammingTutorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Programming Tutor")
        self.root.geometry("800x600")
        
        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)  # Text area will expand
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create text area at top using grid
        self.output_text = scrolledtext.ScrolledText(root, width=70, height=25, font=("Arial", 12))
        self.output_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Create bottom frame for input and buttons
        bottom_frame = tk.Frame(root)
        bottom_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        # Configure bottom frame columns
        bottom_frame.grid_columnconfigure(0, weight=1)  # Input field expands
        bottom_frame.grid_columnconfigure(1, minsize=100)  # Fixed size for buttons
        
        # Input label and entry
        self.label = tk.Label(bottom_frame, text="Ask a question:", font=("Arial", 12))
        self.label.grid(row=0, column=0, sticky="w")
        
        self.input_text = tk.Entry(bottom_frame, width=60, font=("Arial", 12))
        self.input_text.grid(row=1, column=0, sticky="ew", pady=5)

        # Button frame for right-aligned buttons
        button_frame = tk.Frame(bottom_frame)
        button_frame.grid(row=1, column=1, sticky="e", padx=(10, 0))

        self.ask_button = tk.Button(button_frame, text="   Ask   ", command=self.get_answer, font=("Arial", 12), bg="#1F7D53", fg="white")
        self.ask_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(button_frame, text="Clear Chat", command=self.clear_chat, font=("Arial", 12), bg="#ff6666", fg="white")
        self.clear_button.pack(side=tk.LEFT, padx=5)
            
        self.show_welcome_message()

    def show_welcome_message(self):
        welcome_message = """Welcome to the Programming Tutor!
        
        I can help you with questions about C programming. 
        Here are some examples of what you can ask:
        - What is a variable in C?
        - How to print an output?
        - What is dynamic memory allocation?
        - How to create a horizontal tab?

    Type your question below and click 'Ask' to get started.
    """
        self.output_text.insert(tk.END, welcome_message)

    def get_answer(self):
        user_input = self.input_text.get()
        if user_input:
            processed_input = preprocess(user_input)
            
            # Get the predicted answer and its probability
            predicted_probabilities = model.predict_proba([processed_input])[0]
            max_probability = np.max(predicted_probabilities)
            predicted_answer = model.predict([processed_input])[0]
            
            # Set a confidence threshold (e.g., 0.5)
            confidence_threshold = 0.5
            
            if max_probability > confidence_threshold:
                # If the model is confident, display the predicted answer
                self.output_text.insert(tk.END, f"Q: {user_input}\nA: {predicted_answer}\n\n")
            else:
                # If the model is not confident, direct the user to W3Schools
                self.output_text.insert(tk.END, f"Q: {user_input}\nA: I'm not sure about this question. For more details, you can refer to W3 Schools: https://www.w3schools.com/c/index.php\n\n")
            
            self.input_text.delete(0, tk.END)
            # Auto-scroll to bottom
            self.output_text.see(tk.END)
        else:
            self.output_text.insert(tk.END, "Please enter a question.\n\n")
    
    def clear_chat(self):
        """Clear the chat window"""
        self.output_text.delete(1.0, tk.END)
        # Show welcome message again after clearing
        self.show_welcome_message()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ProgrammingTutorApp(root)
    root.mainloop()