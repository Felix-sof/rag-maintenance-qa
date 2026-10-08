---
title: Sensör ve Kolon Referansı
type: reference
source: AI4I 2020 veri seti dokümantasyonu (Matzka, 2020; UCI ML Repository #601)
---

# Sensör ve Kolon Referansı

Bu belge, freze tezgahı log verisindeki her kolonun ne ölçtüğünü ve sentetik veri
setinde nasıl üretildiğini açıklar.

## Ürün Kimliği ve Kalite Tipi (type, product_id)

`type` kolonu ürünün kalite varyantını gösterir: L (düşük), M (orta), H (yüksek).
`product_id`, kalite harfi ile başlayan bir seri numarasıdır. Kalite tipi, aşırı
zorlanma arızası (OSF) sınırını belirler ve takım aşınma hızını etkiler: H, M ve L
tipleri işlenen ürün başına takım aşınmasına sırasıyla 5, 3 ve 2 dakika ekler.

## Hava Sıcaklığı (air_temperature_k)

Kelvin cinsinden ortam hava sıcaklığı. Veri setinde 300 K etrafında, 2 K standart
sapmayla normalize edilmiş bir rastgele yürüyüş (random walk) süreciyle üretilmiştir.
Tek başına bir arıza nedeni değildir, ancak proses sıcaklığıyla arasındaki fark ısı
dağıtım arızasında (HDF) belirleyicidir.

## Proses Sıcaklığı (process_temperature_k)

Kelvin cinsinden proses sıcaklığı. Hava sıcaklığına 10 K eklenerek ve 1 K standart
sapmalı bir rastgele yürüyüş süreciyle üretilmiştir. Bu nedenle hava sıcaklığı ile
proses sıcaklığı arasında güçlü bir pozitif korelasyon beklenir.

## Dönüş Hızı (rotational_speed_rpm)

Dakikadaki devir sayısı. 2860 W'lık bir güçten hesaplanıp üzerine normal dağılımlı
gürültü eklenerek üretilmiştir. Güç sabit tutulmaya çalışıldığı için dönüş hızı ile
tork arasında güçlü bir negatif korelasyon beklenir. Düşük dönüş hızı, ısı dağıtım
arızası (HDF) koşullarından biridir.

## Tork (torque_nm)

Newton metre cinsinden tork. 40 Nm etrafında, 10 Nm standart sapmayla normal
dağılımlı olarak üretilmiş, negatif değerler hariç tutulmuştur. Tork; güç arızası
(PWF, güç = tork × açısal hız) ve aşırı zorlanma arızası (OSF, takım aşınması × tork)
hesaplarına doğrudan girer.

## Takım Aşınması (tool_wear_min)

Kesici takımın dakika cinsinden kullanım süresi. Ürün kalite tipine göre her işlemde
artar (bkz. "Ürün Kimliği ve Kalite Tipi"). Takım aşınması arttıkça TWF ve OSF riski
artar.

## Arıza Etiketleri (machine_failure, twf, hdf, pwf, osf, rnf)

`machine_failure`, herhangi bir arıza modu gerçekleştiğinde 1 olan genel etikettir.
`twf`, `hdf`, `pwf`, `osf` ve `rnf` kolonları ise her bir arıza modunun ayrı ayrı
gerçekleşip gerçekleşmediğini gösterir. Tanımlar için bkz. failure_modes.md.

## Zaman Boyutu Hakkında

Veri setindeki kayıtlar bağımsız proses kayıtlarıdır ve zaman damgası içermez.
Bu nedenle verilerden "bir sonraki arıza ne zaman olur" tahmini yapılamaz; yalnızca
belirli bir sensör değeri kombinasyonu için arıza riski değerlendirilebilir.
