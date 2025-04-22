# word_count_script.py

def count_words_in_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
            words = text.split()
            word_count = len(words)
            return word_count
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None

def main():
    filename = "./words.txt"  # Replace with your actual filename
    word_count = count_words_in_file(filename)

    if word_count is not None:
        print(f"Word count: {word_count}")
        print(f"This file has {word_count} words.")
        words_remaining = 10000 - word_count
        print(f"You have {words_remaining} words left to reach 10,000.")

if __name__ == "__main__":
    main()

