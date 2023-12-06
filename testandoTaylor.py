import cv2
import face_recognition as fr

# Carregar a imagem de referência (imagem do Elon Musk)
imgTaylor = fr.load_image_file('taylor.jpeg')
imgTaylor = cv2.cvtColor(imgTaylor, cv2.COLOR_BGR2RGB)
encodeTaylor = fr.face_encodings(imgTaylor)[0]

# Abrir a webcam
video = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)


if not video.isOpened():
    print("Erro ao abrir a webcam. Verifique a conexão e as permissões.")
else:
    while True:
        # Capturar um frame da webcam
        ret, frame = video.read()

        # Verificar se o frame foi capturado corretamente
        if not ret or frame is None:
            print("Erro ao capturar o frame da webcam.")
            break

        # Converter o frame para RGB (face_recognition utiliza RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Encontrar as localizações dos rostos no frame
        face_locations = fr.face_locations(rgb_frame)

        # Codificar os rostos no frame
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Desenhar um retângulo ao redor do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Comparar a codificação do rosto com a codificação de referência (Elon Musk)
            comparacao = fr.compare_faces([encodeTaylor], face_encoding)
            distancia = fr.face_distance([encodeTaylor], face_encoding)

            # Exibir os resultados no frame
            fonte = cv2.FONT_HERSHEY_DUPLEX
            rotulo = f"Taylor? {comparacao}, Dist: {distancia[0]:.2f}"
            cv2.putText(frame, rotulo, (left, bottom + 20), fonte, 0.6, (255, 255, 255), 1)

        # Exibir o frame
        cv2.imshow('Reconhecimento Facial pela Webcam', frame)

        # Encerrar o loop se 'q' for pressionado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a webcam e fechar a janela
    video.release()
    cv2.destroyAllWindows()
