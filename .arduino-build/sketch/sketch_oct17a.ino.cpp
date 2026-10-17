#include <Arduino.h>
#line 1 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
#include <Wire.h>

// PCA9685 두 대의 I2C 주소입니다.
// 두 번째 보드는 A0 주소 점퍼를 연결하여 0x41로 설정해야 합니다.
const uint8_t PCA1_ADDRESS = 0x40;
const uint8_t PCA2_ADDRESS = 0x41;

const uint8_t SERVO_FREQUENCY = 50;
const uint16_t SERVO_CENTER_US = 1500;  // 일반적인 서보의 90도(중앙) 신호
const uint16_t START_DELAY_MS = 50;

// PCA9685 레지스터 주소
const uint8_t MODE1 = 0x00;
const uint8_t MODE2 = 0x01;
const uint8_t LED0_ON_L = 0x06;
const uint8_t PRESCALE = 0xFE;

#line 18 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
bool devicePresent(uint8_t address);
#line 23 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
bool writeRegister(uint8_t address, uint8_t reg, uint8_t value);
#line 30 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
uint8_t readRegister(uint8_t address, uint8_t reg);
#line 39 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
void setPWMFrequency(uint8_t address, uint8_t frequency);
#line 54 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
bool initializePCA9685(uint8_t address);
#line 66 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
void setPWM(uint8_t address, uint8_t channel, uint16_t on, uint16_t off);
#line 78 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
uint16_t microsecondsToTicks(uint16_t pulseUs);
#line 82 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
void centerAllServos(uint8_t address);
#line 93 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
void setup();
#line 119 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
void loop();
#line 18 "C:\\Users\\User\\Desktop\\ROBOTPAL\\sketch_oct17a\\sketch_oct17a.ino"
bool devicePresent(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

bool writeRegister(uint8_t address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

uint8_t readRegister(uint8_t address, uint8_t reg) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.endTransmission();

  Wire.requestFrom(address, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

void setPWMFrequency(uint8_t address, uint8_t frequency) {
  // PCA9685 내부 발진기 25MHz를 기준으로 prescale 값을 계산합니다.
  const uint8_t prescale =
      (25000000UL / (4096UL * frequency)) - 1;

  const uint8_t oldMode = readRegister(address, MODE1);
  const uint8_t sleepMode = (oldMode & 0x7F) | 0x10;

  writeRegister(address, MODE1, sleepMode);  // 설정 중 발진기 정지
  writeRegister(address, PRESCALE, prescale);
  writeRegister(address, MODE1, oldMode);
  delay(5);
  writeRegister(address, MODE1, oldMode | 0xA0);  // 재시작 + 자동 증가
}

bool initializePCA9685(uint8_t address) {
  if (!devicePresent(address)) {
    return false;
  }

  writeRegister(address, MODE1, 0x00);
  writeRegister(address, MODE2, 0x04);  // 토템폴 출력 방식
  delay(10);
  setPWMFrequency(address, SERVO_FREQUENCY);
  return true;
}

void setPWM(uint8_t address, uint8_t channel, uint16_t on, uint16_t off) {
  const uint8_t firstRegister = LED0_ON_L + (4 * channel);

  Wire.beginTransmission(address);
  Wire.write(firstRegister);
  Wire.write(on & 0xFF);
  Wire.write(on >> 8);
  Wire.write(off & 0xFF);
  Wire.write(off >> 8);
  Wire.endTransmission();
}

uint16_t microsecondsToTicks(uint16_t pulseUs) {
  return (uint32_t)pulseUs * SERVO_FREQUENCY * 4096UL / 1000000UL;
}

void centerAllServos(uint8_t address) {
  const uint16_t centerPulse = microsecondsToTicks(SERVO_CENTER_US);

  for (uint8_t channel = 0; channel < 16; channel++) {
    setPWM(address, channel, 0, centerPulse);

    // 여러 서보의 기동 전류가 한꺼번에 흐르지 않도록 순차적으로 구동합니다.
    delay(START_DELAY_MS);
  }
}

void setup() {
  // Arduino Mega의 I2C 핀: SDA=20, SCL=21
  Wire.begin();
  Serial.begin(9600);

  const bool pca1Ready = initializePCA9685(PCA1_ADDRESS);
  const bool pca2Ready = initializePCA9685(PCA2_ADDRESS);

  if (!pca1Ready) {
    Serial.println("PCA9685 0x40 not found");
  }

  if (!pca2Ready) {
    Serial.println("PCA9685 0x41 not found - check the A0 jumper");
  }

  // PCA9685 두 대에 연결된 전체 32개 채널을 90도로 정렬합니다.
  if (pca1Ready) {
    centerAllServos(PCA1_ADDRESS);
  }

  if (pca2Ready) {
    centerAllServos(PCA2_ADDRESS);
  }
}

void loop() {
  // PCA9685가 마지막 PWM 신호를 계속 출력하므로 추가 동작은 필요하지 않습니다.
}

