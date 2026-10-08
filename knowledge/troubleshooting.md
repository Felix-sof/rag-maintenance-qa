---
title: Sorun Giderme Kılavuzu
type: example
source: Örnek amaçlı hazırlanmış sorun giderme kılavuzu (gerçek bir üreticiye ait değildir)
priority: 1
updated: 2026-10-08
---

# Sorun Giderme Kılavuzu

Bu kılavuz, sensör verisinde görülen belirtilerden olası nedenlere ve önerilen
aksiyonlara giden bir başlangıç noktasıdır. Veri analizi bir belirtiyi gösterdiğinde
(ör. anomali, arızalı kayıtlarda farklılaşan bir sensör), ilgili başlıktaki kontroller
uygulanmalıdır.

## Belirti: Tork Değerleri Yüksek

Olası nedenler:

- Kesici takım aşınmış veya körelmiş.
- Kesme parametreleri (ilerleme, kesme derinliği) malzeme için fazla agresif.
- Talaş tahliyesi yetersiz, talaş sıkışıyor.
- Soğutma ve yağlama yetersiz.

Risk: yüksek tork ile yüksek takım aşınmasının birleşimi aşırı zorlanma arızasına
(OSF) yol açar; yüksek tork ile yüksek devrin birleşimi ise güç sınırını aşarak güç
arızasına (PWF) neden olabilir.

Önerilen aksiyonlar: takım durumunu kontrol edin ve gerekirse değiştirin
(maintenance_procedures.md > Kesici Takım Değişimi); kesme parametrelerini gözden
geçirin; talaş tahliyesini ve soğutmayı kontrol edin.

## Belirti: Tork Değerleri Düşük veya Sıfıra Yakın

Olası nedenler: takım kırılmış veya iş parçasıyla temas yok; sensör veya bağlantı
arızası; boşta çalışma kayıtlarının üretim verisine karışması.

Risk: düşük tork düşük devirle birleşirse proses gücü alt sınırın altına düşer ve
güç arızası (PWF) oluşabilir.

Önerilen aksiyonlar: takımı görsel olarak kontrol edin; tork sensörünün kalibrasyonunu
doğrulayın; boşta çalışma kayıtlarını filtreleyin.

## Belirti: Dönüş Hızı Düşük

Olası nedenler: yüksek tork altında motorun yavaşlaması; sürücü parametre hatası;
kayış kayması.

Risk: düşük devir, sıcaklık farkının da düşük olduğu durumlarda ısı dağıtım
arızasına (HDF) yol açar.

Önerilen aksiyonlar: iş mili ve tahrik sistemi kontrolü uygulayın
(maintenance_procedures.md > İş Mili ve Tahrik Sistemi Kontrolü); soğutmayı
kontrol edin.

## Belirti: Proses ve Hava Sıcaklığı Farkı Azalıyor

Olası nedenler: soğutma sistemi verimi düşmüş; ortam sıcaklığı yükselmiş; fan veya
hava kanalları tıkalı.

Risk: düşük devirle birlikte ısı dağıtım arızası (HDF) koşulunu oluşturur.

Önerilen aksiyonlar: soğutma sistemi kontrolü uygulayın
(maintenance_procedures.md > Soğutma Sistemi Kontrolü).

## Belirti: Takım Aşınması Yüksek Değerlere Ulaşıyor

Olası nedenler: planlı takım değişimi aksatılmış; aşınma sayacı sıfırlanmamış;
yüksek kaliteli (H) ürünlerde aşınma daha hızlı birikir.

Risk: takım aşınması arızası (TWF) ve, tork da yüksekse, aşırı zorlanma arızası (OSF).

Önerilen aksiyonlar: takımı değiştirin; planlı değişim aralığını arıza verisine göre
yeniden belirleyin; aşınma sayacının değişimlerde sıfırlandığını doğrulayın.

## Belirti: Sensör Değerinde Tekil Uç Değer (Anomali)

Olası nedenler: gerçek bir proses olayı (takım kırılması, sıkışma); sensör gürültüsü
veya iletişim hatası; veri girişi ya da birim dönüşümü hatası.

Önerilen aksiyonlar: uç değerin bir arıza etiketiyle eş zamanlı olup olmadığını
kontrol edin; aynı kayıttaki diğer sensörlerin de sapıp sapmadığına bakın; tek
sensörde izole bir sapma varsa önce sensör ve veri hattını doğrulayın.
