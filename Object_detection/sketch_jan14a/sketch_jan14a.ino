

int yellowLed=A1;
int redLed = A0;

String data  = "";



void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(yellowLed,OUTPUT);
  pinMode(redLed,OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:

  while(Serial.available() == 0){

  }

  data = Serial.readStringUntil('\n');

  if(data == "Person"){
    digitalWrite(yellowLed,HIGH);
    digitalWrite(redLed,LOW);
  }
  else{
    digitalWrite(yellowLed,LOW);
    digitalWrite(redLed,HIGH);
  }
  

}
