import cv2
from deepface import DeepFace

# Kamera açma
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera hatası!")
        break

    try:
        # Yüz analizi yap (enforce_detection=False)
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Sonuçları ekranda göster
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"DeepFace hatası: {e}")

    # Görüntüyü ekranda göster
    cv2.imshow('Emotion Detection', frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
