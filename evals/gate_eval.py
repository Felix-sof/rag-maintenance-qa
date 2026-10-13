"""
Off-topic gate eval - no LLM, no API key, no quota.

Before any LLM call, rag.py refuses a question whose best retrieved chunk is
too weak a match. Two signals could carry that decision: the embedding
cosine, or the cross-encoder's rerank score. This measures both on three
kinds of question:

    in_scope     the documents answer it - refusing it is the costly mistake
    near_domain  maintenance, but not in the documents (another machine, live
                 plant data, purchasing) - a refusal here saves an LLM call
    off_topic    unrelated

    python -m evals.gate_eval            # compare the two gates
    python -m evals.gate_eval --detail   # every question with both scores
    python -m evals.gate_eval --check    # CI gate: exit 1 if the pipeline's gate
                                         # (rag.is_off_topic) refuses an in-scope question

Each kind has two sets. CALIBRATION is what the thresholds were chosen on
(its in-scope questions are the retrieval eval's). HOLDOUT was written after
MIN_RERANK_SCORE was fixed and before its scores were seen, to check that
the threshold wasn't fitted to one set of questions.

The retriever always runs in hybrid_rerank mode here, so both scores exist
for every question.
"""

import sys

import rag
from evals.retrieval_eval import EASY, HARD
from retriever import Retriever

CALIBRATION = {
    "in_scope": [question for question, _ in EASY + HARD],
    "near_domain": [
        "Hidrolik presin yağ değişimi nasıl yapılır?",
        "Bu ay fabrikada kaç arıza oldu?",
        "Torna tezgahında ayna nasıl sökülür?",
        "Kompresörün hava filtresi ne sıklıkla değiştirilir?",
        "İş mili rulmanları hangi gresle yağlanır?",
        "Soğutma sıvısı hangi marka olmalı?",
        "Tezgahın garanti süresi ne kadar?",
        "Kaynak makinesinde elektrot nasıl değiştirilir?",
        "G-kodu ile dairesel interpolasyon nasıl programlanır?",
        "Forklift akü bakımı nasıl yapılır?",
        "Titreşim sensörünün alarm eşiği kaç mm/s?",
        "Konveyör bandının gerginliği nasıl ayarlanır?",
        "Dün gece vardiyasında hangi tezgah durdu?",
        "Yedek parça siparişi kime verilir?",
        "Enjeksiyon makinesinde kalıp sıcaklığı kaç derece olmalı?",
        "Operatörlerin hangi sertifikaya sahip olması gerekir?",
        "Elektrik panosundaki sigorta değerleri nedir?",
        "How do I change the oil in the hydraulic press?",
        "What is the lubrication interval for the spindle bearings?",
        "Which operator was on shift when the last failure happened?",
    ],
    "off_topic": [
        "En iyi pizza tarifi nedir?",
        "Türkiye'nin başkenti neresi?",
        "Python'da liste nasıl sıralanır?",
        "Yarın İstanbul'da hava nasıl olacak?",
        "Galatasaray maçı kaç kaç bitti?",
        "Bana bir film önerir misin?",
        "Dolar kuru bugün ne kadar?",
        "Kedim neden sürekli miyavlıyor?",
        "İngilizce öğrenmek için hangi kitabı okumalıyım?",
        "Kahve nasıl demlenir?",
        "Merhaba, nasılsın?",
        "2 + 2 kaç eder?",
        "What is the capital of France?",
        "Write a poem about the sea",
        "How do I reset my email password?",
    ],
}

HOLDOUT = {
    "in_scope": [
        "PWF",
        "HDF nedir?",
        "Ana şalteri kapattıktan sonra ne yapmalıyım?",
        "Gözlük takmak zorunlu mu?",
        "Nozullar tıkanmış olabilir, nasıl kontrol ederim?",
        "Aylık bakımda hangi ölçümler yapılır?",
        "M tipi ürünler için OSF sınırı kaç?",
        "Proses sıcaklığı verisi nasıl üretilmiş?",
        "Tork ortalaması kaç Nm?",
        "Takım değişiminden sonra deneme kesimi gerekli mi?",
        "Fan kanalları tozlanınca ne olur?",
        "Etiketi kim çıkarabilir?",
        "machine_failure kolonu ne zaman 1 olur?",
        "Güç 9000 W'ı geçerse ne olur?",
        "Koruyucu ekipman",
        "Why does torque drop when speed increases?",
        "When should the coolant system be checked?",
        "tool wear failure threshold in minutes",
        "Sürücü hata kayıtlarına ne zaman bakılır?",
        "Talaş sıkışması torku etkiler mi?",
        "Veri setinde zaman damgası var mı?",
        "Düşük devirde uzun süre çalışmak gerekirse ne önerilir?",
        "RNF işaretliyken makine arızası etiketi neden 0 olabilir?",
        "Kilitleme",
        "Tezgah gövdesi aşırı ısınıyor",
    ],
    "near_domain": [
        "Pres kalıbı nasıl değiştirilir?",
        "Lazer kesim makinesinde lens temizliği nasıl yapılır?",
        "Geçen hafta en çok hangi arıza modu görüldü?",
        "İş mili motorunu hangi firma üretiyor?",
        "Soğutma pompasının yedek parça numarası nedir?",
        "Bakım ekibinin vardiya çizelgesi nedir?",
        "Robot kolun kalibrasyonu nasıl yapılır?",
        "Tezgahın satın alma fiyatı ne kadar?",
        "Hava kompresöründe basınç şalteri nasıl ayarlanır?",
        "How often should the compressor oil be changed?",
        "Taşlama tezgahında taş nasıl bilenir?",
        "Yangın söndürücüler nerede duruyor?",
    ],
    "off_topic": [
        "İstanbul'dan Ankara'ya kaç saat sürer?",
        "Bugün günlerden ne?",
        "Bana bir fıkra anlat",
        "Bitcoin fiyatı ne oldu?",
        "En iyi dizüstü bilgisayar hangisi?",
        "How tall is Mount Everest?",
        "Makarna kaç dakika haşlanır?",
        "Translate 'hello' into German",
    ],
}

SETS = {"calibration": CALIBRATION, "holdout": HOLDOUT}
KINDS = ("in_scope", "near_domain", "off_topic")

# gate name -> does it refuse a question with these scores?
GATES = {
    f"cosine < {rag.MIN_SCORE}": lambda row: row["cosine"] < rag.MIN_SCORE,
    f"rerank < {rag.MIN_RERANK_SCORE}": lambda row: row["rerank"] < rag.MIN_RERANK_SCORE,
}


def measure(retriever, top_k: int = rag.DEFAULT_TOP_K) -> dict:
    """Best cosine and best rerank score of every question, plus the pipeline's own verdict."""
    results = {}
    for set_name, kinds in SETS.items():
        results[set_name] = {}
        for kind in KINDS:
            rows = []
            for question in kinds[kind]:
                hits = retriever.search(question, top_k=top_k)
                rows.append(
                    {
                        "question": question,
                        "cosine": max(h["score"] for h in hits),
                        "rerank": max(h["ranks"]["rerank"] for h in hits),
                        "refused": rag.is_off_topic(hits),
                    }
                )
            results[set_name][kind] = rows
    return results


if __name__ == "__main__":
    results = measure(Retriever(mode="hybrid_rerank"))

    if "--detail" in sys.argv:
        for set_name, kinds in results.items():
            for kind, rows in kinds.items():
                print(f"\n--- {set_name} / {kind} ({len(rows)} questions) ---")
                for row in sorted(rows, key=lambda r: r["rerank"]):
                    mark = "REFUSED" if row["refused"] else ""
                    print(f"rerank {row['rerank']:7.2f}   cosine {row['cosine']:.3f}   {mark:8s}{row['question']}")

    print("\nQuestions refused before the LLM (in_scope should be 0, the others as many as possible)\n")
    print(f"{'gate':16s} | {'calibration':^38s} | {'holdout':^38s}")
    print(f"{'':16s} | " + " | ".join(" ".join(f"{kind:>12s}" for kind in KINDS) for _ in SETS))
    print("-" * 98)
    for name, refuses in GATES.items():
        cells = []
        for kinds in results.values():
            counts = [f"{sum(refuses(row) for row in kinds[kind])}/{len(kinds[kind])}" for kind in KINDS]
            cells.append(" ".join(f"{count:>12s}" for count in counts))
        print(f"{name:16s} | " + " | ".join(cells))

    if "--check" in sys.argv:
        wrongly_refused = [
            row["question"] for kinds in results.values() for row in kinds["in_scope"] if row["refused"]
        ]
        if wrongly_refused:
            print(f"\nFAIL: the gate refuses {len(wrongly_refused)} in-scope question(s): {wrongly_refused}")
            sys.exit(1)
        print("\nOK: the gate refuses no in-scope question")
