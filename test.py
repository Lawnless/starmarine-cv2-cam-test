import cv2 # OpenCV kütüphanesini projemize dahil ettik.
import math # math kütüphanesini projemize dahil ettik.
import numpy as np # OpenCV'nin olmazsa olmazı numpy'ı dahil ettik.

kontorler = {} # Bütün kontörleri dizi olarak tanıtıyoruz.
koseler = [] # Poligon köşelerini dizi olarak tanıtıyoruz.
kamera = cv2.VideoCapture(0) # Kamerayı tanıtıyoruz.
print("'Q' tuşuna basarak kapatabilirsiniz.")

# Videoyu XVID codecinde .avi uzantısında kaydetmek için birkaç kod.
vcodec = cv2.VideoWriter_fourcc(*'XVID')
cikti = cv2.VideoWriter('cikti.avi', vcodec, 20.0, (640, 480))

def aciHesapla(pt1, pt2, pt0): # Açı hesaplama fonksiyonumuz.
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

while (kamera.isOpened()): # Kamera açık olduğu sürece aşağıdaki kodu oynatıyoruz.
    # Kamerayı her bir framede tekrar oku. (takip et)
    ret, frame = kamera.read()
    if ret == True:
        # Görüntüyü griyle filtrele.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Canny'yi ayarla.
        canny = cv2.Canny(frame,80,240,3)

        # Kontörleri bulup döngüye aldık, her biri için kontrol yapacağız.
        kontorler, siralama = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(kontorler)):
            # Sonuçları kesin doğru yapmak için poligonların köşelerine bakıyoruz.
            koseler = cv2.approxPolyDP(kontorler[i], cv2.arcLength(kontorler[i], True) * 0.02, True)

            # Ufak veya belirlenemeyen objeleri atla.
            if (abs(cv2.contourArea(kontorler[i])) < 100 or not(cv2.isContourConvex(koseler))):
                continue

            # Üçgen
            if (len(koseler) == 3):
                x,y,w,h = cv2.boundingRect(kontorler[i])
                cv2.putText(frame,'TRI',(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            elif (len(koseler) >= 4 and len(koseler) <= 6):
                # Poligonal nesnenin etrafını kontrol edelim.
                vtc = len(koseler)
                # Bütün köşelerin cos derecelerini alalım.
                cos = []
                for j in range(2,vtc+1):
                    cos.append(aciHesapla(koseler[j%vtc], koseler[j-2], koseler[j-1]))
                # cos dereceleri sırala.
                cos.sort()
                # En düşük ve en yüksek açılara bak.
                mincos = cos[0]
                maxcos = cos[-1]

                # Köşelerini sayıp derecesini hesapla ve algıla.
                x,y,w,h = cv2.boundingRect(kontorler[i])
                if(vtc==4):
                    cv2.putText(frame,'KARE',(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
                elif(vtc==5):
                    cv2.putText(frame,'BESGEN',(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
                elif(vtc==6):
                    cv2.putText(frame,'ALTIGEN',(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            else:
                # Daireyi algıla ve işaretle.
                area = cv2.contourArea(kontorler[i])
                x,y,w,h = cv2.boundingRect(kontorler[i])
                radius = w/2
                if(abs(1 - (float(w)/h))<=2 and abs(1-(area/(math.pi*radius*radius)))<=0.2):
                    cv2.putText(frame,'DAIRE',(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)

        # Sonuçu gösterme ve .avi dosyasına yazma kısmı.
        cikti.write(frame)
        cv2.imshow('Kamera', frame)
        cv2.imshow('Kamera 2', canny)
        if cv2.waitKey(1) == 1048689:
            break

# Her şey bittikten sonra kamerayı kapatabiliriz.
kamera.release()
cv2.destroyAllWindows()
