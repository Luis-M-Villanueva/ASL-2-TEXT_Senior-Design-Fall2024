#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <HTTPClient.h>
#include "esp_http_server.h"
#include "img_converters.h"
#include <WiFi.h>
#include <ArduCAM.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

/** PROG VARS **/

const char* ssid = "Sulis";
const char* pass = "ESP32prj!";
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace; boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n"; 

httpd_handle_t stream_httpd = NULL;
bool capcap = false;
const int CS = 10;
ArduCAM myCAM(OV2640, CS);

uint8_t* current_buf = (uint8_t *)malloc(MAX_FIFO_SIZE);

TaskHandle_t capStart;
TaskHandle_t httpServe;

httpd_req_t *current_req = NULL; 
// Semaphore for synchronization
SemaphoreHandle_t frame_ready;

static esp_err_t stream_handler(httpd_req_t *req);
void startCameraServer();
void captureTask(void *pvParameters);
void sendTask(void *pvParameters);

void setup() {
  // Set up camera and WiFi (as per original code)
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

  // Initialize camera module
  while (1) {
    myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
    temp = myCAM.read_reg(ARDUCHIP_TEST1);
    if (temp != 0x55){
      Serial.println(F("## SPI interface Error! "));
      delay(1000);
      continue;
    } else {
      Serial.println(F("## CMD SPI interface OK. "));
      break;
    }
  }

  while (1) {
    myCAM.wrSensorReg8_8(0xff, 0x01);
    myCAM.rdSensorReg8_8(OV2640_CHIPID_HIGH, &vid);
    myCAM.rdSensorReg8_8(OV2640_CHIPID_LOW, &pid);
    if ((vid != 0x26 ) && (( pid != 0x41 ) || ( pid != 0x42 ))) {
      Serial.println(F("## Can't find OV2640 module!"));
      delay(1000);
      continue;
    } else {
      Serial.println(F("## OV2640 detected ")); 
      break;
    }
  }

  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_352x288);

  // Initialize semaphore
  frame_ready = xSemaphoreCreateBinary();

  // Connect to WiFi
  WiFi.begin(ssid, pass);
  Serial.print("Waiting for WiFi to connect...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(200);
    Serial.print(".");
  }
  Serial.println(" connected");

  // Start camera server
  startCameraServer();

  // Create FreeRTOS tasks for capturing and sending
  xTaskCreatePinnedToCore(captureTask, "Capture Task", 30000, NULL, 1, &capStart, 1); // Capture on core 1
  xTaskCreatePinnedToCore(sendTask, "Send Task", 30000, NULL, 1, &httpServe, 0); // Send on core 0

  Serial.print("Camera Ready Use ==> 'http://");
  Serial.print(WiFi.localIP());
  Serial.println(":80/video_stream");
}

void loop() {
  // Main loop does nothing; tasks run on different cores
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  
  httpd_uri_t index_uri = {
    .uri       = "/video_stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };
  
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
    Serial.println("HTTP SERVER STARTED");
  } else {
    Serial.println("HTTP SERVER FAILED");
  }
}

// Capture Task: Captures frames into the buffer
void captureTask(void *pvParameters) {
  uint8_t temp;
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
      uint8_t *buf_to_fill = (uint8_t *)malloc(length);
      if (!buf_to_fill) {
        Serial.println("Memory allocation failed!");
      }

      myCAM.CS_LOW();
      myCAM.set_fifo_burst();
      for (uint32_t i = 0; i < length; i++) {
        buf_to_fill[i] = SPI.transfer(0x00);
      }
      myCAM.CS_HIGH();

      // Signal that the frame is ready to be sent
      memcpy(current_buf,buf_to_fill,length);
      xSemaphoreGive(frame_ready);
      
    }
    capcap = true;
  }
}

// Send Task: Sends frames to the HTTP client
void sendTask(void *pvParameters) {
  esp_err_t res = ESP_OK;
  char part_buf[64];
  httpd_req_t *req = current_req;
  while (true) {
    // Wait for the capture task to finish capturing a frame
    length = sizeof(current_buf);
    if (req == NULL) {
      break;
    } 
    if (xSemaphoreTake(frame_ready, portMAX_DELAY) == pdTRUE) {
      Serial.println("GONNA DO IT");
      res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
      if (res != ESP_OK) {
        Serial.println("Failed to set content type.");
        break;
      }

      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART,length);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
      
      if (res == ESP_OK) {
        res = httpd_resp_send_chunk(req, (const char *)current_buf,length);
      }

      if (res == ESP_OK) {
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
      }
      if (res != ESP_OK) {
        Serial.println("Failed to send chunk.");
        continue;
      }
      
    }
  }
  xSemaphoreGive()
  myCAM.clear_fifo_flag();
}

static esp_err_t stream_handler(httpd_req_t *req) {
  current_req = req;  // Store the HTTP request
  Serial.println("STREAM HANDLER");
  return ESP_OK;  // The actual sending is handled by sendTask
}
