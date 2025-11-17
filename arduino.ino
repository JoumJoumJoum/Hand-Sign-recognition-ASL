#include <LiquidCrystal.h>

// RS=7, EN=8, D4=9, D5=10, D6=11, D7=12
LiquidCrystal lcd(7, 8, 9, 10, 11, 12);
String incoming = "";

void setup() {
  lcd.begin(16, 2);
  lcd.print("Waiting...");
  Serial.begin(9600);  // must match Python
}

void loop() {
  if (Serial.available()) {
    incoming = Serial.readStringUntil('\n');
    incoming.trim();

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Sign:");
    lcd.setCursor(0, 1);
    lcd.print(incoming);
  }
}
