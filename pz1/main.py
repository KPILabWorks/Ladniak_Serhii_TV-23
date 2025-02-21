import csv

def read_csv_to_dicts(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

# Приклад використання
if __name__ == "__main__":
    file_path = "example.csv"  # Вкажіть ваш шлях до файлу
    data = read_csv_to_dicts(file_path)
    for row in data:
        print(row)
