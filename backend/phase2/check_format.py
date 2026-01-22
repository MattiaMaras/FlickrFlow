import os

#Script creato perchè data cleaning trovava un solo record nel dataset

def check_file_structure():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "..", "data", "flickr2x.json")

    print(f"Ispeziono i primi caratteri di: {file_path}")

    if not os.path.exists(file_path):
        print("File non trovato!")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        # Lettura dei primi 500 caratteri
        content = f.read(500)
        print("\n--- INIZIO DEL FILE ---")
        print(content)
        print("\n--- FINE ANTEPRIMA ---")

        # Check diagnostico
        if content.strip().startswith("["):
            print("\nDIAGNOSI: Il file è un JSON ARRAY (inizia con [).")
        elif content.strip().startswith("{"):
            print("\n DIAGNOSI: Il file è JSON LINES (inizia con {).")
        else:
            print("\n️ DIAGNOSI: Formato sconosciuto o sporco.")


if __name__ == "__main__":
    check_file_structure()