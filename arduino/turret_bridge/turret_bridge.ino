/*
 * turret_bridge.ino
 * =================
 * Arduino firmware for the pan/tilt turret + laser.
 *
 * Hardware connections (ASSUMPTION - adjust to your wiring):
 *   Pin 9  --> Pan servo signal wire  (horizontal)
 *   Pin 10 --> Tilt servo signal wire (vertical)
 *   Pin 7  --> Laser module IN pin    (HIGH = ON)
 *   GND    --> Common ground for servos + laser
 *   5V     --> Servo VCC (or use external 5V/6V for high-torque servos)
 *
 * Serial protocol (115200 baud):
 *   P<float>\n  -- set pan  servo angle 0-180 degrees
 *   T<float>\n  -- set tilt servo angle 0-180 degrees
 *   L1\n        -- laser ON
 *   L0\n        -- laser OFF
 *
 * Example:
 *   "P90.0\n"  --> pan servo to 90 degrees (centre)
 *   "T65.0\n"  --> tilt servo to 65 degrees
 *   "L1\n"     --> laser on
 */

#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PIN_PAN   = 9;
const int PIN_TILT  = 10;
const int PIN_LASER = 7;

void setup() {
    Serial.begin(115200);
    panServo.attach(PIN_PAN);
    tiltServo.attach(PIN_TILT);
    pinMode(PIN_LASER, OUTPUT);
    digitalWrite(PIN_LASER, LOW);

    // Initialise to centre position
    panServo.write(90);
    tiltServo.write(90);

    Serial.println("READY");
}

void loop() {
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) return;

        char cmd = line.charAt(0);
        String val = line.substring(1);

        if (cmd == 'P') {
            float angle = val.toFloat();
            angle = constrain(angle, 0.0, 180.0);
            panServo.write((int)angle);
            Serial.print("PAN:");
            Serial.println(angle);

        } else if (cmd == 'T') {
            float angle = val.toFloat();
            angle = constrain(angle, 0.0, 180.0);
            tiltServo.write((int)angle);
            Serial.print("TILT:");
            Serial.println(angle);

        } else if (cmd == 'L') {
            if (val == "1") {
                digitalWrite(PIN_LASER, HIGH);
                Serial.println("LASER:ON");
            } else {
                digitalWrite(PIN_LASER, LOW);
                Serial.println("LASER:OFF");
            }
        }
    }
}
