set export CYC_HOST=localhost
set export CYC_PORT=3601
set CYC_BRIDGE_HTTP_PORT=8081
set "JAVA8=C:\Program Files\Eclipse Adoptium\jdk-8.0.472.8-hotspot\bin\java.exe"

"%JAVA8%" -jar target/cyc-bridge-server.jar
