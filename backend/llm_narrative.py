import os
import requests


from langchain_core.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM
from langchain_ollama import ChatOllama


def init_environment():
    # Setup Java 17
    jdk_path = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = f"{jdk_path}/bin:" + os.environ.get("PATH", "")


def check_ollama_status():
    """Health Check del server AI locale"""
    try:
        response = requests.get("http://localhost:11434", timeout=1)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def get_hybrid_chain(prompt_template):
    """Factory Pattern: Sceglie tra AI Reale (Mistral) e Mock (Backup)"""

    # CASO A: AI REALE (LOCALE)
    if check_ollama_status():
        print("Server Ollama attivo. Inferenza locale con modello 'Mistral'.")
        # temperature=0.6 per un bilanciamento tra creatività e coerenza
        llm_real = ChatOllama(model="mistral", temperature=0.6)
        return prompt_template | llm_real

    # CASO B: MOCK (FALLBACK)
    else:
        print("AI Locale non disponibile. Attivazione Backup Mock.")
        simulated_response = (
            "📄 REPORT STRATEGICO (SISTEMA DI BACKUP):\n\n"
            "Dall'analisi dei flussi emerge chiaramente che il Vaticano funge da 'Sink Node' (punto di assorbimento finale) "
            "per il 93.7% dei turisti che visitano il Centro Storico.\n"
            "Proposta Operativa: Si suggerisce l'introduzione immediata del biglietto integrato 'Roma Imperiale-Vaticano' "
            "e l'istituzione di una navetta dedicata dai Fori Imperiali a San Pietro per decongestionare la rete pubblica."
        )
        llm_fake = FakeListLLM(responses=[simulated_response])
        return prompt_template | llm_fake


def run_genai_narrative():
    init_environment()

    print("\n" + "=" * 60)
    print("FASE 7: GENERATIVE AI REPORTING")
    print("=" * 60)
    print("Obiettivo: Generazione report strategico con AI Locale (Mistral).")

    # Dati Reali (Input), presi dalla fase 6, trajectory_mining.py
    top_rule = {
        "antecedent": "Trastevere, Piazza di Spagna, Piazza Navona, Fontana di Trevi, Colosseo",
        "consequent": "Vaticano",
        "confidence": "93.7%",
        "lift": "1.80"
    }

    top_itemset = {
        "items": "Colosseo, Vaticano",
        "count": "5301 visitatori"
    }

    # --- 2. PROMPT ENGINEERING ---
    template = """
    Agisci come un esperto stratega turistico per il Comune di Roma.
    Hai ricevuto i seguenti dati dal sistema di Big Data Analytics:

    1. Pattern di Massa: Il percorso "{itemset_items}" è stato effettuato da {itemset_count} persone.
    2. Regola Forte: Chi visita "{rule_antecedent}" ha il {rule_confidence} di probabilità di finire il tour al {rule_consequent}.

    TASK:
    Scrivi una breve nota strategica (massimo 5 righe) indirizzata al Sindaco.
    - Spiega in italiano fluido e professionale che il {rule_consequent} è il punto di arrivo naturale dei flussi.
    - Proponi una soluzione concreta (es. trasporti o bigliettazione) per gestire questo spostamento di massa.
    - Non usare saluti generici, vai dritto al punto.
    """

    prompt = PromptTemplate(
        input_variables=["itemset_items", "itemset_count", "rule_antecedent", "rule_consequent", "rule_confidence",
                         "rule_lift"],
        template=template
    )

    try:
        chain = get_hybrid_chain(prompt)

        print("⏳ Analisi e generazione del testo in corso...")

        inputs = {
            "itemset_items": top_itemset["items"],
            "itemset_count": top_itemset["count"],
            "rule_antecedent": top_rule["antecedent"],
            "rule_consequent": top_rule["consequent"],
            "rule_confidence": top_rule["confidence"],
            "rule_lift": top_rule["lift"]
        }

        response = chain.invoke(inputs)
        final_text = response.content if hasattr(response, 'content') else str(response)

        print("\n📄 REPORT GENERATO:")
        print("-" * 60)
        print(final_text.strip())  # .strip() rimuove spazi vuoti inutili all'inizio/fine
        print("-" * 60 + "\n")

    except Exception as e:
        print(f"❌ Errore pipeline: {e}")


if __name__ == "__main__":
    run_genai_narrative()