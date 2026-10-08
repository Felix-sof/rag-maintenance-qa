---
title: Arıza Modları ve Oluşma Koşulları
type: reference
source: AI4I 2020 veri seti dokümantasyonu (Matzka, 2020; UCI ML Repository #601)
---

# Arıza Modları ve Oluşma Koşulları

AI4I 2020 veri setinde `machine_failure` etiketi, aşağıdaki beş bağımsız arıza
modundan en az biri gerçekleştiğinde 1 olur. Bir kayıtta birden fazla arıza modu
aynı anda işaretlenmiş olabilir. Arıza modları sentetik veri üretimi sırasında
aşağıdaki kurallarla tanımlanmıştır; yani bu koşullar, gerçek bir tezgahta
"yaklaşık" değil, bu veri setinde arızayı tetikleyen kesin kurallardır.

## TWF - Takım Aşınması Arızası (Tool Wear Failure)

Kesici takım, takım aşınma süresi 200 ile 240 dakika arasında rastgele seçilen bir
noktaya ulaştığında ya değiştirilir ya da kırılır. Bu nedenle TWF, takım aşınması
(`tool_wear_min`) yüksek değerlere çıktıkça ortaya çıkar; düşük aşınmada TWF
beklenmez.

Önemli not: veri setinde takımın değiştirildiği durumlar ile gerçekten kırıldığı
durumlar aynı TWF etiketiyle işaretlenir. Bu yüzden TWF etiketi her zaman fiziksel
bir kırılma anlamına gelmez.

İlgili prosedür: bkz. "Kesici Takım Değişimi" (maintenance_procedures.md).

## HDF - Isı Dağıtım Arızası (Heat Dissipation Failure)

Proses ısısı yeterince dağıtılamadığında oluşur. Tetiklenme koşulu: hava sıcaklığı
ile proses sıcaklığı arasındaki fark 8,6 K'nin altına düşer **ve** aynı anda dönüş
hızı 1380 rpm'nin altındadır. Her iki koşul birlikte sağlanmalıdır.

Yorum: düşük dönüş hızında soğutucu hava akışı azalır; ortam havası ile proses
arasındaki sıcaklık farkı da küçükse ısı tezgahtan uzaklaştırılamaz.

İlgili prosedür: bkz. "Soğutma Sistemi Kontrolü" (maintenance_procedures.md).

## PWF - Güç Arızası (Power Failure)

Proses için gereken güç, tork ile dönüş hızının (rad/s cinsinden) çarpımıdır:
güç [W] = tork [Nm] × dönüş hızı [rad/s]. Dönüş hızı rpm'den rad/s'ye
rpm × 2π / 60 ile çevrilir. Güç 3500 W'ın altına düşerse veya 9000 W'ı aşarsa
proses başarısız olur.

Yorum: PWF hem çok düşük güçte (ör. düşük tork ve düşük hız) hem de çok yüksek
güçte (ör. yüksek tork ve yüksek hız) oluşabilir; tek yönlü bir eşik değildir.

## OSF - Aşırı Zorlanma Arızası (Overstrain Failure)

Takım aşınması ile torkun çarpımı (dakika × Nm) ürün kalite tipine göre belirlenen
sınırı aşarsa oluşur:

- L (düşük kalite) tipi: 11.000 minNm
- M (orta kalite) tipi: 12.000 minNm
- H (yüksek kalite) tipi: 13.000 minNm

Yorum: aşınmış bir takımla yüksek torkta çalışmak, özellikle L tipi ürünlerde
OSF riskini artırır. Takım aşınması arttıkça güvenli tork sınırı düşer.

## RNF - Rastgele Arıza (Random Failure)

Her proseste, proses parametrelerinden bağımsız olarak %0,1 olasılıkla rastgele
arıza oluşur. RNF sensör değerleriyle açıklanamaz; bu nedenle sensör verisinden
tahmin edilmesi beklenmez.

Not: veri setinde RNF işaretli bazı kayıtlarda `machine_failure` etiketi 0 olabilir;
bu veri setinin bilinen bir tutarsızlığıdır.
