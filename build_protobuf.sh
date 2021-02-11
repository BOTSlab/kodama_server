# Compiles protobuf message classes to c++ and python directories.
protoc -I=./ --cpp_out=./src ./kodama_msg.proto
protoc -I=./ --python_out=./for_python_client ./kodama_msg.proto