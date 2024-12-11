import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cvzone

cap = cv2.VideoCapture(0)  # Kamera başlatma
cap.set(3, 1280)  # Genişlik ayarı
cap.set(4, 720)   # Yükseklik ayarı
detector = HandDetector(detectionCon=0.8)  # El algılama ayarı
colorR = (255, 0, 255)  # Renk ayarı

# Sürüklenebilir dikdörtgen sınıfı
class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        # Dikdörtgeni sürüklemek için koşul
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

rectList = []
# 5 adet sürüklenebilir dikdörtgen oluşturma
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()  # Kamera görüntüsünü al
    if not success:
        print("Kamera açılmadı!")
        break  # Kameraya bağlanılamazsa döngüden çık

    img = cv2.flip(img, 1)  # Görüntüyü yatayda çevir
    hands, img = detector.findHands(img)  # El algılama

    if hands:  # El algılandığında
        lmList = hands[0]['lmList']  # Parmak noktalarını al
        x1, y1, z1 = lmList[8]  # Baş parmak
        x2, y2, z2 = lmList[12]  # İşaret parmağı

        # İki parmak arasındaki mesafeyi ölçme
        l, _, _ = detector.findDistance([x1, y1], [x2, y2], img)
        print(f"Mesafe: {l}")

        # Mesafe 40'a kadar hassasiyet düşer
        if l < 40:
            cursor = [x2, y2]  # İşaret parmağının koordinatları
            for rect in rectList:
                rect.update(cursor)  # Dikdörtgenleri güncelle

    imgNew = np.zeros_like(img, np.uint8)  # Yeni bir görüntü oluştur
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        # Dikdörtgeni çizme
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()  # Orijinal görüntüyü kopyala
    alpha = 0.1
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]  # Maskeyi uygula

    cv2.imshow("Image", out)  # Görüntüyü göster

    # "q" tuşuna basarak çıkış yapabilirsiniz
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
