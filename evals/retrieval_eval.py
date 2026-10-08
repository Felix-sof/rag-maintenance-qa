"""
Retrieval-only eval - no LLM, no API key, no quota.

The answer eval (answer_eval.py) checks the final answers, but it can't
tell whether a wrong answer came from bad retrieval or from the model
ignoring good retrieval. This isolates the retriever: for each question, is
the section that actually answers it among the top-k results?

    python -m evals.retrieval_eval                  # compare all retrieval modes
    python -m evals.retrieval_eval --mode hybrid    # one mode, with per-question detail
    python -m evals.retrieval_eval --min-mrr 0.9    # CI gate: exit 1 if the default mode
                                                    # (RETRIEVAL_MODE) scores below this on any set

Two question sets:
    EASY  phrased close to the documents' own wording (written alongside them)
    HARD  paraphrases that share few words with the answer ("makine ısınıp
          yavaş dönüyor" -> HDF), bare codes and numbers ("OSF", "8,6 K"),
          acronyms the documents never spell out ("LOTO", "KKD"), and English
          questions against Turkish documents

Metrics: hit@1, hit@4 (share of questions whose expected section is in the
top 1 / top 4) and MRR (mean reciprocal rank; 1.0 = always first). A case
passes if ANY of its expected sections is retrieved; matching is a substring
match on "<source> > <section>".
"""

import sys

from retriever import MODES, RETRIEVAL_MODE, Retriever

# (question, [expected "source > section" substrings])
# EASY: phrased close to the documents' own wording.
EASY = [
    ("HDF arızası hangi koşullarda oluşur?", ["failure_modes.md > Arıza Modları ve Oluşma Koşulları > HDF"]),
    ("Isı dağıtım arızası için devir ve sıcaklık eşiği nedir?", ["failure_modes.md > Arıza Modları ve Oluşma Koşulları > HDF"]),
    ("Güç arızası nasıl hesaplanır?", ["failure_modes.md > Arıza Modları ve Oluşma Koşulları > PWF"]),
    ("Overstrain failure limit for L type products", ["failure_modes.md > Arıza Modları ve Oluşma Koşulları > OSF"]),
    ("TWF etiketi her zaman takımın kırıldığı anlamına mı gelir?", ["failure_modes.md > Arıza Modları ve Oluşma Koşulları > TWF"]),
    ("Rastgele arızalar sensörlerden tahmin edilebilir mi?", ["failure_modes.md > Arıza Modları ve Oluşma Koşulları > RNF"]),
    ("Kesici takım nasıl değiştirilir, adımlar neler?", ["maintenance_procedures.md > Bakım Prosedürleri > Kesici Takım Değişimi"]),
    ("Soğutma sistemini nasıl kontrol ederim?", ["maintenance_procedures.md > Bakım Prosedürleri > Soğutma Sistemi Kontrolü"]),
    ("Motor sürücüsü ve iş mili kontrolü", ["maintenance_procedures.md > Bakım Prosedürleri > İş Mili ve Tahrik Sistemi Kontrolü"]),
    ("Haftalık bakımda neler yapılmalı?", ["maintenance_procedures.md > Bakım Prosedürleri > Periyodik Bakım Planı"]),
    ("Tork değerleri sürekli yüksek çıkıyor, sebebi ne olabilir?", ["troubleshooting.md > Sorun Giderme Kılavuzu > Belirti: Tork Değerleri Yüksek"]),
    ("Tork sıfıra yakın ölçülüyor", ["troubleshooting.md > Sorun Giderme Kılavuzu > Belirti: Tork Değerleri Düşük"]),
    ("Proses sıcaklığı ile hava sıcaklığı arasındaki fark azalıyor", [
        "troubleshooting.md > Sorun Giderme Kılavuzu > Belirti: Proses ve Hava Sıcaklığı Farkı",
        "failure_modes.md > Arıza Modları ve Oluşma Koşulları > HDF",
    ]),
    ("Bir sensörde tek bir uç değer gördüm, ne yapmalıyım?", ["troubleshooting.md > Sorun Giderme Kılavuzu > Belirti: Sensör Değerinde Tekil Uç Değer"]),
    ("Bakımdan önce kilitleme etiketleme nasıl yapılır?", ["safety.md > Bakım Güvenliği > Enerji İzolasyonu"]),
    ("Dönen iş mili yakınında eldiven takılır mı?", ["safety.md > Bakım Güvenliği > Kişisel Koruyucu Donanım"]),
    ("Torque kolonu neyi ölçer, nasıl üretilmiş?", ["sensor_reference.md > Sensör ve Kolon Referansı > Tork"]),
    ("Kalite tipi takım aşınmasını nasıl etkiler?", ["sensor_reference.md > Sensör ve Kolon Referansı > Ürün Kimliği ve Kalite Tipi"]),
    ("Dönüş hızı ile tork arasında neden negatif ilişki var?", [
        "sensor_reference.md > Sensör ve Kolon Referansı > Dönüş Hızı",
        "sensor_reference.md > Sensör ve Kolon Referansı > Tork",
    ]),
    ("Verilerden bir sonraki arızanın zamanı tahmin edilebilir mi?", ["sensor_reference.md > Sensör ve Kolon Referansı > Zaman Boyutu"]),
]


# HARD: written to stress the retriever, not to match the documents' wording.
HARD = [
    # bare codes and numbers - exact tokens, little meaning for an embedding
    ("OSF", ["> OSF"]),
    ("TWF", ["> TWF"]),
    ("PWF sınır değerleri", ["> PWF"]),
    ("RNF olasılığı", ["> RNF"]),
    ("13.000 minNm hangi ürün tipi için geçerli?", ["> OSF"]),
    ("8,6 K sıcaklık farkı neyi ifade eder?", ["> HDF"]),
    ("2860 W güç değeri nereden geliyor?", ["> Dönüş Hızı"]),
    # acronyms the documents never spell out
    ("LOTO prosedürü", ["Enerji İzolasyonu"]),
    ("KKD listesi", ["Kişisel Koruyucu Donanım"]),
    # paraphrases with little word overlap
    ("Makine çok ısınıyor ve yavaş dönüyorsa hangi arıza tetiklenir?", ["> HDF", "Belirti: Dönüş Hızı Düşük"]),
    ("Kesici uç körelmişken yüksek kuvvetle çalışmanın riski nedir?", ["> OSF", "Belirti: Tork Değerleri Yüksek"]),
    ("Motorun çektiği güç çok düşükse proses neden başarısız olur?", ["> PWF"]),
    ("Kesici uç kaç dakika kullanımdan sonra kırılır?", ["> TWF"]),
    ("Sebepsiz, rastgele bozulma ihtimali var mı?", ["> RNF"]),
    ("Elektriği kesip asma kilit takmak", ["Enerji İzolasyonu"]),
    ("İşlemden yeni çıkmış parçaya elle dokunabilir miyim?", ["Sıcak Yüzeyler"]),
    ("Kırılan uç parçalarını nasıl toplarım?", ["Sıcak Yüzeyler"]),
    ("Hangi ürün kalitesi aleti en hızlı aşındırır?", ["Ürün Kimliği", "Belirti: Takım Aşınması Yüksek"]),
    ("Ortam havası kaç Kelvin civarında?", ["> Hava Sıcaklığı"]),
    ("Devir artınca moment neden düşüyor?", ["> Dönüş Hızı", "> Tork"]),
    ("Yıllık revizyonda neler yapılır?", ["Periyodik Bakım Planı"]),
    ("Rulmandan ses geliyor", ["İş Mili ve Tahrik Sistemi"]),
    ("Soğutma sıvısı konsantrasyonu", ["Soğutma Sistemi Kontrolü"]),
    ("Sensörde tek seferlik ani sıçrama gördüm", ["Tekil Uç Değer"]),
    ("Aşınma sayacı sıfırlanmıyor", ["Belirti: Takım Aşınması Yüksek", "Kesici Takım Değişimi"]),
    ("Kayış kayıyor olabilir mi?", ["Belirti: Dönüş Hızı Düşük", "İş Mili ve Tahrik Sistemi"]),
    ("Boşta çalışma kayıtları veriyi bozar mı?", ["Belirti: Tork Değerleri Düşük"]),
    # English questions, Turkish documents
    ("How do I replace the cutting tool?", ["Kesici Takım Değişimi"]),
    ("What protective equipment is required?", ["Kişisel Koruyucu Donanım"]),
    ("heat dissipation failure conditions", ["> HDF"]),
]

SETS = {"easy": EASY, "hard": HARD}


def evaluate(retriever, cases, k: int = 4) -> dict:
    rows = []
    for question, expected in cases:
        labels = [f"{h['source']} > {h['section']}" for h in retriever.search(question, top_k=k)]
        rank = next((i + 1 for i, label in enumerate(labels) if any(e in label for e in expected)), None)
        rows.append({"question": question, "rank": rank, "top": labels[:2]})
    n = len(rows)
    return {
        "hit@1": sum(r["rank"] == 1 for r in rows) / n,
        f"hit@{k}": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "rows": rows,
    }


def compare(k: int = 4) -> dict:
    results = {}
    for mode in MODES:
        retriever = Retriever(mode=mode)
        results[mode] = {name: evaluate(retriever, cases, k) for name, cases in SETS.items()}
    return results


if __name__ == "__main__":
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 4

    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
        retriever = Retriever(mode=mode)
        for name, cases in SETS.items():
            report = evaluate(retriever, cases, k)
            print(f"\n--- {name} ({len(cases)} questions) ---")
            for row in report["rows"]:
                print(f"{('#' + str(row['rank'])) if row['rank'] else 'MISS':5s} {row['question']}")
                if row["rank"] != 1:
                    print(f"       top: {row['top']}")
            print(f"hit@1 {report['hit@1']:.0%}   hit@{k} {report[f'hit@{k}']:.0%}   MRR {report['mrr']:.3f}")
        sys.exit(0)

    results = compare(k)
    print(f"\n{'mode':15s} | {'easy (' + str(len(EASY)) + ')':^26s} | {'hard (' + str(len(HARD)) + ')':^26s}")
    print(f"{'':15s} | {'hit@1':>7s} {'hit@' + str(k):>7s} {'MRR':>8s}  | {'hit@1':>7s} {'hit@' + str(k):>7s} {'MRR':>8s}")
    print("-" * 74)
    for mode, by_set in results.items():
        cells = "  | ".join(
            f"{r['hit@1']:7.0%} {r[f'hit@{k}']:7.0%} {r['mrr']:8.3f}" for r in (by_set["easy"], by_set["hard"])
        )
        print(f"{mode:15s} | {cells}")

    if "--min-mrr" in sys.argv:
        threshold = float(sys.argv[sys.argv.index("--min-mrr") + 1])
        below = {name: r["mrr"] for name, r in results[RETRIEVAL_MODE].items() if r["mrr"] < threshold}
        if below:
            print(f"\nFAIL: {RETRIEVAL_MODE} MRR below {threshold}: {below}")
            sys.exit(1)
        print(f"\nOK: {RETRIEVAL_MODE} MRR >= {threshold} on every set")
