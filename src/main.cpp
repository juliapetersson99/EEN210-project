#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Adafruit_BusIO_Register.h>

const int MPU6050_ADDR = 0x68; 

uint8_t readRegister(uint8_t reg) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false); 
  Wire.requestFrom(MPU6050_ADDR, (uint8_t)1);
  if (Wire.available()) {
    return Wire.read();
  }
  return 0;
}

void writeRegister(uint8_t reg, uint8_t data) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission();
}

// Public internet, allow port in firewall
// Replace with your network credentials
const char *ssid = "asuss iPhone";
const char *password = "klolp123"; 

const int PWR_MGMT_1 = 0x6B;
const int CONFIG = 0x1A; 
const int GYRO_CONFIG = 0x1B;
const int ACCEL_CONFIG = 0x1C;

// Replace with your WebSocket server address
const char *webSocketServer = "172.20.10.2";

const int webSocketPort = 8000; 
const char *webSocketPath = "/";
MPU6050 mpu; // Define the sensor
WebSocketsClient client;
// SocketIOClient socketIO;

bool wifiConnected = false;

void setup()
{
  Serial.begin(9600);
  Wire.begin();
  mpu.initialize();

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());
  client.begin(webSocketServer, webSocketPort, "/ws");
  Serial.println(client.isConnected());
  wifiConnected = true;

  uint8_t accel_config = readRegister(ACCEL_CONFIG);
  accel_config &= ~0x18; // Clear bits 4:3
  accel_config |= (2 << 3); // Set to 2 for ±8g
  uint8_t accel_range = (accel_config & 0x18) >> 3; // Bits 4:3
 
  writeRegister(ACCEL_CONFIG, accel_config);
  Serial.print("Accelerometer Range: ");
  switch (accel_range) {
    case 0: Serial.println("±2g"); break;
    case 1: Serial.println("±4g"); break;
    case 2: Serial.println("±8g"); break;
    case 3: Serial.println("±16g"); break;
    default: Serial.println("Unknown");
  }
  
  

  uint8_t gyro_config = readRegister(GYRO_CONFIG);
  writeRegister(GYRO_CONFIG, gyro_config);
  gyro_config &= ~0x18; // Clear bits 4:3 (FS_SEL)
  gyro_config |= (3 << 3);
  uint8_t gyro_range = (gyro_config & 0x18) >> 3; // Bits 4:3
  Serial.print("Gyroscope Range: ");
  switch (gyro_range) {
    case 0: Serial.println("±250°/s"); break;
    case 1: Serial.println("±500°/s"); break;
    case 2: Serial.println("±1000°/s"); break;
    case 3: Serial.println("±2000°/s"); break;
    default: Serial.println("Unknown");
  }
  delay(1000);



  //


}

void loop()
{
  if (wifiConnected)
  {
    client.loop();

    // Get sensor data
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);


    // Convert data to a JSON string
    String payload = "{\"acceleration_x\":" + String(ax) +
                     ",\"acceleration_y\":" + String(ay) +
                     ",\"acceleration_z\":" + String(az) +
                     ",\"gyroscope_x\":" + String(gx) +
                     ",\"gyroscope_y\":" + String(gy) +
                     ",\"gyroscope_z\":" + String(gz) + "}";

    Serial.println("Skickar....");
    // server address, port and URL
    // Send data via WebSocket
    client.sendTXT(payload);
    client.loop();

    delay(10); // Adjust delay as needed
  }
}
