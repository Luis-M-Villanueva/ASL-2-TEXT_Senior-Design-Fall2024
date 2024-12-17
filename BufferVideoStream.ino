/*
  ASL2SPEECH 

  ESP32-S3-WROOM2-N32R8V-DevkitC1

  Desc:
  ESP32 is setup to use ArduCAM OV2640 Mini 2MP Plus to send video/images to an http server the esp is connect to through WiFi. 

  Required
    -Show ouput to Audio or LCD screen
    -Establish an initialization and termination to start and finish
    -Test with video
    -Integrate i.e. establish camera connection to model
    -Final integration of CAM to ESP to Server to Audio/Visual output

*/

#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <HTTPClient.h>
#include "esp_http_server.h"
#include "img_converters.h"
#include <WiFi.h>
#include <ArduCAM.h>

/** PROG VARS **/

const char* ssid = "Sulis";
const char* pass = "ESP32prj!";
uint8_t mode = 0;
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace; boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n"; 

httpd_handle_t stream_httpd = NULL;

bool capcap = false;
bool is_header = false;
const int CS = 10;

ArduCAM myCAM(OV2640,CS);

/* FUNCTION INIT*/

static esp_err_t stream_handler(httpd_req_t *req);
void startCameraServer();

void setup() {

  uint8_t vid, pid;
  uint8_t temp; 
  Wire.begin();
  Serial.begin(2000000);
  
  Serial.println(F("## ArduCAM Start! "));
  pinMode(CS,OUTPUT);
  digitalWrite(CS,HIGH);
  SPI.begin();
  SPI.beginTransaction(SPISettings(8000000,MSBFIRST,SPI_MODE0));
  myCAM.write_reg(0x07, 0x80);
  myCAM.write_reg(0x07, 0x00);

  while(1){
    //Check if the ArduCAM SPI bus is OK
    myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
    temp = myCAM.read_reg(ARDUCHIP_TEST1);
    if (temp != 0x55){
      Serial.println(F("## SPI interface Error! "));
      //delay(1);
      continue;
    }else{
      Serial.println(F("## CMD SPI interface OK. "));
      break;
    }
  }

  while (1) {
    //Check if the camera module type is OV2640
    myCAM.wrSensorReg8_8(0xff, 0x01);
    myCAM.rdSensorReg8_8(OV2640_CHIPID_HIGH, &vid);
    myCAM.rdSensorReg8_8(OV2640_CHIPID_LOW, &pid);
    if ((vid != 0x26 ) && (( pid != 0x41 ) || ( pid != 0x42 ))) {
      Serial.println(F("## Can't find OV2640 module!"));
      //delay(1); 
      continue;
    }
    else {
      Serial.println(F("## OV2640 detected ")); 
      break;
    }
  }
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_352x288);
  myCAM.OV2640_set_Contrast(Contrast1);  
  myCAM.OV2640_set_Brightness(Brightness1);
  myCAM.OV2640_set_Color_Saturation(Saturation1);
  myCAM.OV2640_set_Light_Mode(Home);
  myCAM.OV2640_set_Special_effects(Normal);

  Serial.println(F("## Settings Adjusted and Cam Init. "));
  myCAM.clear_fifo_flag();

    /* WiFi & Server */ 
  WiFi.begin(ssid, pass);
  Serial.print("Waiting for WiFi to connect...");
  while ((WiFi.status() != WL_CONNECTED)) {
    delay(200);
    Serial.print(".");
  }
  Serial.print("");
  Serial.println(" connected");
  

    // Init Cam Server
  startCameraServer();  
  Serial.print("Camera Ready Use ==> 'http://");
  Serial.print(WiFi.localIP());
  Serial.print(":80/video_stream");
  Serial.println("");
}

void loop() {
  
}

void startCameraServer(){
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port=80;
  httpd_uri_t index_uri = {
    .uri       = "/video_stream",
    .method    = HTTP_GET,
    .handler   = cap_handler,
    .user_ctx  = NULL
  };
  //Serial.printf("Starting web server on port: '%d'\n", config.server_port);
 
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
    Serial.println("HTTP SERVER STARTED");
  }
  else
  {
    Serial.println("HTTP SERVER FAILED");
  }
}

static esp_err_t cap_handler(httpd_req_t *req) {
  esp_err_t res = ESP_OK;
  char part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }

  while (true) {
    if (capcap) {
      myCAM.flush_fifo();
      myCAM.clear_fifo_flag();
      myCAM.start_capture();
      capcap = false;
    }

    if (myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
      uint32_t length = myCAM.read_fifo_length();
      if (length == 0 || length > MAX_FIFO_SIZE) {
        myCAM.clear_fifo_flag();
        continue;
      }

      uint8_t *buf = (uint8_t *)malloc(length);
      if (!buf) {
        Serial.println("Memory allocation failed!");
        return ESP_ERR_NO_MEM;
      }

      myCAM.CS_LOW();
      myCAM.set_fifo_burst();
      for (uint32_t i = 0; i < length; i++) {
        buf[i] = SPI.transfer(0x00);
      }
      myCAM.CS_HIGH();

      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, length);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
      if (res == ESP_OK) {
        res = httpd_resp_send_chunk(req, (const char *)buf, length);
      }
      if (res == ESP_OK) {
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
      }

      if (res != ESP_OK) {
        break;
      }
    }
    
    myCAM.clear_fifo_flag();
    capcap = true;
  }

  return res;
}


