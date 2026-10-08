"""
Retrieval-only eval (vector_store.py) - no LLM, no API key, no quota.

The answer eval (answer_eval.py) checks the final answers, but it can't
tell whether a wrong answer came from bad retrieval or from the model
ignoring good retrieval. This isolates the retriever: for each question, is
the section that actually answers it among the top-k results?

Usage:
    python -m evals.retrieval_eval          # k = 4 (rag.DEFAULT_TOP_K)
    python -m evals.retrieval_eval --k 2

Metrics:
    hit@k  share of questions whose expected section is in the top k
    MRR    mean reciprocal rank of the expected section (1.0 = always first)

A case passes if ANY of its expected sections is retrieved (some questions
are legitimately answered by more than one section). Matching is a
substring match on "<source> > <section>", so a case can name a whole
document ("troubleshooting.md") or one section ("HDF").
"""

import sys

from vector_store import get_store

# (question, [expected "source > section" substrings])
CASES = [
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


def evaluate(k: int = 4) -> dict:
    store = get_store()
    store.build()
    rows = []
    for question, expected in CASES:
        hits = store.search(question, top_k=k)
        labels = [f"{h['source']} > {h['section']}" for h in hits]
        rank = next(
            (i + 1 for i, label in enumerate(labels) if any(e in label for e in expected)),
            None,
        )
        rows.append({"question": question, "rank": rank, "top": labels[:2]})

    hit_rate = sum(r["rank"] is not None for r in rows) / len(rows)
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows)
    return {"k": k, "hit_rate": hit_rate, "mrr": mrr, "rows": rows, "model": store.embedder.name}


if __name__ == "__main__":
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 4
    report = evaluate(k)
    for row in report["rows"]:
        marker = f"#{row['rank']}" if row["rank"] else "MISS"
        print(f"{marker:5s} {row['question']}")
        if row["rank"] != 1:
            print(f"       top: {row['top']}")
    print("\n" + "=" * 60)
    print(f"Embedding model: {report['model']}")
    print(f"hit@{k}: {report['hit_rate']:.0%}   MRR: {report['mrr']:.3f}   ({len(report['rows'])} questions)")
