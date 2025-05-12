#!/bin/bash

# --- CONFIGURAZIONE ---
EXECUTABLE_PATH="CUDA/build/main"  # Modifica con il percorso corretto del tuo eseguibile
REP=10                        # Numero di ripetizioni per ogni configurazione di N
K_VALUE=1000                  # Valore fisso per K
F_VALUE=1                     # Valore fisso per f (sempre 1)
OUTPUT_CSV="cuda_profiling_results_raw.csv"

# Valori di N da testare (esempio: 2^16, 2^17, 2^18, 2^19, 2^20)
# Puoi modificarli come preferisci
N_VALUES=()
for i in {16..21}; do
  N_VALUES+=($((2**i)))
done
# Oppure definiscili manualmente: N_VALUES=(65536 131072 262144)

# --- CONTROLLI PRELIMINARI ---
if [ ! -x "$EXECUTABLE_PATH" ]; then
  echo "Errore: Eseguibile '$EXECUTABLE_PATH' non trovato o non eseguibile."
  exit 1
fi

# --- INTESTAZIONE DEL CSV ---
# L'intestazione ora include anche un identificatore per la singola esecuzione (run)
echo "N,K,RunNumber,TimeShm,ThroughputShm,TimeNoShm,ThroughputNoShm,Correctness" > "$OUTPUT_CSV"

# --- CICLO PRINCIPALE DI PROFILING ---
for N_CURRENT in "${N_VALUES[@]}"; do
  echo "----------------------------------------------------"
  echo "Inizio profiling per N = $N_CURRENT, K = $K_VALUE (REP = $REP)"

  for ((r=1; r<=REP; r++)); do
    echo "  Esecuzione $r/$REP per N=$N_CURRENT..."
    # Esegui il programma e cattura l'output
    # Aggiungiamo un timeout per evitare blocchi indefiniti (es. 300 secondi = 5 minuti)
    # output_programma=$(timeout 300s "$EXECUTABLE_PATH" "$N_CURRENT" "$K_VALUE" "$F_VALUE")
    # Se non hai `timeout` o non vuoi usarlo:
    output_programma=$("$EXECUTABLE_PATH" "$N_CURRENT" "$K_VALUE" "$F_VALUE")
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "    ATTENZIONE: L'eseguibile ha restituito un codice di errore $exit_code per N=$N_CURRENT, run $r. Output: '$output_programma'. Salto questa esecuzione."
        # Opzionale: scrivere una riga con NA se si vuole tracciare il fallimento nel CSV
        # echo "$N_CURRENT,$K_VALUE,$r,NA,NA,NA,NA,NA" >> "$OUTPUT_CSV"
        continue # Salta al prossimo ciclo di REP
    fi

    if [ -z "$output_programma" ]; then
        echo "    ATTENZIONE: Output vuoto dall'eseguibile per N=$N_CURRENT, run $r. Salto questa esecuzione."
        # Opzionale: scrivere una riga con NA
        # echo "$N_CURRENT,$K_VALUE,$r,NA,NA,NA,NA,NA" >> "$OUTPUT_CSV"
        continue
    fi

    # Controlla se l'output ha il formato atteso (5 campi separati da virgola)
    # `awk` è generalmente disponibile, se non lo fosse, questo controllo diventerebbe più complesso
    num_fields=$(echo "$output_programma" | awk -F',' '{print NF}')
    if [ "$num_fields" -ne 5 ]; then
        echo "    ATTENZIONE: Formato output non valido per N=$N_CURRENT, run $r. Output: '$output_programma'. Attesi 5 campi, trovati $num_fields. Salto questa esecuzione."
        # Opzionale: scrivere una riga con NA
        # echo "$N_CURRENT,$K_VALUE,$r,NA,NA,NA,NA,NA" >> "$OUTPUT_CSV"
        continue
    fi

    # Scrivi i risultati direttamente nel file CSV
    # L'output del programma è già formattato con virgole, quindi lo aggiungiamo direttamente
    echo "$N_CURRENT,$K_VALUE,$r,$output_programma" >> "$OUTPUT_CSV"

    echo "    Risultati esecuzione $r per N=$N_CURRENT salvati."
  done
  echo "  Completate $REP esecuzioni per N=$N_CURRENT."
done

echo "----------------------------------------------------"
echo "Profiling completato. I risultati grezzi di ogni esecuzione sono in $OUTPUT_CSV"