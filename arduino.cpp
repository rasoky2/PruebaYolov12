#include <Servo.h>

Servo miServo;
bool contaminado = false;
String comando = "";

// Ángulos
const int POS_BASE = 0;
const int POS_ACTIVA = 45;

// Retención en 45° tras activar (ms)
const unsigned long ACTIVO_MS = 1500;  // 1.5 s
unsigned long activoHastaMs = 0;

int anguloActual = POS_BASE;

void moverServoA(int angulo) {
  if (angulo != anguloActual) {
    miServo.write(angulo);
    anguloActual = angulo;
  }
}

void setup() {
  Serial.begin(9600);
  miServo.attach(9);
  moverServoA(POS_BASE);  // Posición inicial
  Serial.println("Arduino listo");
}

void loop() {
  // Leer comandos desde Python
  if (Serial.available()) {
    comando = Serial.readStringUntil('\n');
    comando.trim();

    if (comando == "CONTAMINADO") {
      contaminado = true;
      activoHastaMs = millis() + ACTIVO_MS;  // mantener activo por ventana
      moverServoA(POS_ACTIVA); // Ir a 45°
      Serial.println("CONTAMINADO: servo en 45° (retención)");
    } else if (comando == "SANO") {
      contaminado = false;
      moverServoA(POS_BASE);   // Abrir inmediatamente
      Serial.println("SANO: servo en 0°");
    }
  }

  // Lógica de retención y retorno automático
  if (contaminado) {
    if (millis() <= activoHastaMs) {
      moverServoA(POS_ACTIVA); // Mantener 45° durante la ventana
    } else {
      contaminado = false;     // Termina ciclo
      moverServoA(POS_BASE);   // Volver a base automáticamente
      Serial.println("Fin retención: servo en 0°");
    }
  } else {
    moverServoA(POS_BASE);     // Asegurar base
  }
}
