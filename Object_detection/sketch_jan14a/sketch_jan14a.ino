

#include <Servo.h>
Servo servo;

int yellowLed=A1;
int redLed = A0;
int lazerPin = A2;

String data  = "";
String servoData = "";

int pos = 0;



void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(yellowLed,OUTPUT);
  pinMode(redLed,OUTPUT); 

  servo.attach(D12);

}

void loop() {
  // put your main code here, to run repeatedly:

  while(Serial.available() == 0){

  }

  data = Serial.readStringUntil('\n');

  if(data != "Not"){
    if(data == "po")
        pos = 180;
    else
      pos = 0;
    digitalWrite(yellowLed,HIGH);
    digitalWrite(lazerPin,HIGH);
    digitalWrite(redLed,LOW);

    servo.write(pos);

  }
  else{
    digitalWrite(yellowLed,LOW);
    digitalWrite(redLed,HIGH);
    digitalWrite(lazerPin,LOW);
    servo.write(90);
  }
  

}
