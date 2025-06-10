import cv2
import mediapipe as mp

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mediapipe hand setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Fingertip landmark indexes
tip_ids = [4, 8, 12, 16, 20]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Flip image horizontally for correct hand side
    img = cv2.flip(img, 1)

    # FACE DETECTION
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # HAND DETECTION
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            fingers = []
            if lm_list:
                # Thumb
                if lm_list[tip_ids[0]][0] < lm_list[tip_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                # Other 4 fingers
                for i in range(1, 5):
                    fingers.append(1 if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1] else 0)

            total_fingers = sum(fingers)

            # Gesture recognition
            gesture = "Unknown"
            if total_fingers == 0:
                gesture = "Fist"
            elif total_fingers == 1:
                gesture = "1"
            elif total_fingers == 2:
                gesture = "2"
            elif total_fingers == 3:
                gesture = "3"
            elif total_fingers == 4:
                gesture = "4"
            elif total_fingers == 5:
                gesture = "Open Hand"
            label = handedness.classification[0].label
            corrected_label = "right" if label == "left" else "left"

            cv2.putText(img, f'{corrected_label} Hand: {gesture}',
                        (10 + 220 * idx, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show result
    cv2.imshow("Face + Hand + Gesture", img)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
