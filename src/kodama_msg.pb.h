// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: kodama_msg.proto

#ifndef PROTOBUF_INCLUDED_kodama_5fmsg_2eproto
#define PROTOBUF_INCLUDED_kodama_5fmsg_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3007000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3007001 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/timestamp.pb.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_kodama_5fmsg_2eproto

// Internal implementation detail -- do not use these members.
struct TableStruct_kodama_5fmsg_2eproto {
  static const ::google::protobuf::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::ParseTable schema[4]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors_kodama_5fmsg_2eproto();
namespace kodama {
class RequestData;
class RequestDataDefaultTypeInternal;
extern RequestDataDefaultTypeInternal _RequestData_default_instance_;
class SensorData;
class SensorDataDefaultTypeInternal;
extern SensorDataDefaultTypeInternal _SensorData_default_instance_;
class SensorData_Pose2D;
class SensorData_Pose2DDefaultTypeInternal;
extern SensorData_Pose2DDefaultTypeInternal _SensorData_Pose2D_default_instance_;
class SensorData_Position2D;
class SensorData_Position2DDefaultTypeInternal;
extern SensorData_Position2DDefaultTypeInternal _SensorData_Position2D_default_instance_;
}  // namespace kodama
namespace google {
namespace protobuf {
template<> ::kodama::RequestData* Arena::CreateMaybeMessage<::kodama::RequestData>(Arena*);
template<> ::kodama::SensorData* Arena::CreateMaybeMessage<::kodama::SensorData>(Arena*);
template<> ::kodama::SensorData_Pose2D* Arena::CreateMaybeMessage<::kodama::SensorData_Pose2D>(Arena*);
template<> ::kodama::SensorData_Position2D* Arena::CreateMaybeMessage<::kodama::SensorData_Position2D>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace kodama {

// ===================================================================

class RequestData :
    public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:kodama.RequestData) */ {
 public:
  RequestData();
  virtual ~RequestData();

  RequestData(const RequestData& from);

  inline RequestData& operator=(const RequestData& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  RequestData(RequestData&& from) noexcept
    : RequestData() {
    *this = ::std::move(from);
  }

  inline RequestData& operator=(RequestData&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor() {
    return default_instance().GetDescriptor();
  }
  static const RequestData& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const RequestData* internal_default_instance() {
    return reinterpret_cast<const RequestData*>(
               &_RequestData_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(RequestData* other);
  friend void swap(RequestData& a, RequestData& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline RequestData* New() const final {
    return CreateMaybeMessage<RequestData>(nullptr);
  }

  RequestData* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<RequestData>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const RequestData& from);
  void MergeFrom(const RequestData& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  static const char* _InternalParse(const char* begin, const char* end, void* object, ::google::protobuf::internal::ParseContext* ctx);
  ::google::protobuf::internal::ParseFunc _ParseFunc() const final { return _InternalParse; }
  #else
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(RequestData* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // int32 tag_id = 1;
  void clear_tag_id();
  static const int kTagIdFieldNumber = 1;
  ::google::protobuf::int32 tag_id() const;
  void set_tag_id(::google::protobuf::int32 value);

  // int32 v = 2;
  void clear_v();
  static const int kVFieldNumber = 2;
  ::google::protobuf::int32 v() const;
  void set_v(::google::protobuf::int32 value);

  // int32 w = 3;
  void clear_w();
  static const int kWFieldNumber = 3;
  ::google::protobuf::int32 w() const;
  void set_w(::google::protobuf::int32 value);

  // int32 tau = 4;
  void clear_tau();
  static const int kTauFieldNumber = 4;
  ::google::protobuf::int32 tau() const;
  void set_tau(::google::protobuf::int32 value);

  // int32 scenario = 6;
  void clear_scenario();
  static const int kScenarioFieldNumber = 6;
  ::google::protobuf::int32 scenario() const;
  void set_scenario(::google::protobuf::int32 value);

  // int32 targetX = 7;
  void clear_targetx();
  static const int kTargetXFieldNumber = 7;
  ::google::protobuf::int32 targetx() const;
  void set_targetx(::google::protobuf::int32 value);

  // int32 targetY = 8;
  void clear_targety();
  static const int kTargetYFieldNumber = 8;
  ::google::protobuf::int32 targety() const;
  void set_targety(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:kodama.RequestData)
 private:
  class HasBitSetters;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::int32 tag_id_;
  ::google::protobuf::int32 v_;
  ::google::protobuf::int32 w_;
  ::google::protobuf::int32 tau_;
  ::google::protobuf::int32 scenario_;
  ::google::protobuf::int32 targetx_;
  ::google::protobuf::int32 targety_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_kodama_5fmsg_2eproto;
};
// -------------------------------------------------------------------

class SensorData_Position2D :
    public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:kodama.SensorData.Position2D) */ {
 public:
  SensorData_Position2D();
  virtual ~SensorData_Position2D();

  SensorData_Position2D(const SensorData_Position2D& from);

  inline SensorData_Position2D& operator=(const SensorData_Position2D& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  SensorData_Position2D(SensorData_Position2D&& from) noexcept
    : SensorData_Position2D() {
    *this = ::std::move(from);
  }

  inline SensorData_Position2D& operator=(SensorData_Position2D&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor() {
    return default_instance().GetDescriptor();
  }
  static const SensorData_Position2D& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const SensorData_Position2D* internal_default_instance() {
    return reinterpret_cast<const SensorData_Position2D*>(
               &_SensorData_Position2D_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void Swap(SensorData_Position2D* other);
  friend void swap(SensorData_Position2D& a, SensorData_Position2D& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline SensorData_Position2D* New() const final {
    return CreateMaybeMessage<SensorData_Position2D>(nullptr);
  }

  SensorData_Position2D* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<SensorData_Position2D>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const SensorData_Position2D& from);
  void MergeFrom(const SensorData_Position2D& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  static const char* _InternalParse(const char* begin, const char* end, void* object, ::google::protobuf::internal::ParseContext* ctx);
  ::google::protobuf::internal::ParseFunc _ParseFunc() const final { return _InternalParse; }
  #else
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(SensorData_Position2D* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // int32 x = 1;
  void clear_x();
  static const int kXFieldNumber = 1;
  ::google::protobuf::int32 x() const;
  void set_x(::google::protobuf::int32 value);

  // int32 y = 2;
  void clear_y();
  static const int kYFieldNumber = 2;
  ::google::protobuf::int32 y() const;
  void set_y(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:kodama.SensorData.Position2D)
 private:
  class HasBitSetters;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::int32 x_;
  ::google::protobuf::int32 y_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_kodama_5fmsg_2eproto;
};
// -------------------------------------------------------------------

class SensorData_Pose2D :
    public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:kodama.SensorData.Pose2D) */ {
 public:
  SensorData_Pose2D();
  virtual ~SensorData_Pose2D();

  SensorData_Pose2D(const SensorData_Pose2D& from);

  inline SensorData_Pose2D& operator=(const SensorData_Pose2D& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  SensorData_Pose2D(SensorData_Pose2D&& from) noexcept
    : SensorData_Pose2D() {
    *this = ::std::move(from);
  }

  inline SensorData_Pose2D& operator=(SensorData_Pose2D&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor() {
    return default_instance().GetDescriptor();
  }
  static const SensorData_Pose2D& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const SensorData_Pose2D* internal_default_instance() {
    return reinterpret_cast<const SensorData_Pose2D*>(
               &_SensorData_Pose2D_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  void Swap(SensorData_Pose2D* other);
  friend void swap(SensorData_Pose2D& a, SensorData_Pose2D& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline SensorData_Pose2D* New() const final {
    return CreateMaybeMessage<SensorData_Pose2D>(nullptr);
  }

  SensorData_Pose2D* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<SensorData_Pose2D>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const SensorData_Pose2D& from);
  void MergeFrom(const SensorData_Pose2D& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  static const char* _InternalParse(const char* begin, const char* end, void* object, ::google::protobuf::internal::ParseContext* ctx);
  ::google::protobuf::internal::ParseFunc _ParseFunc() const final { return _InternalParse; }
  #else
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(SensorData_Pose2D* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // int32 x = 1;
  void clear_x();
  static const int kXFieldNumber = 1;
  ::google::protobuf::int32 x() const;
  void set_x(::google::protobuf::int32 value);

  // int32 y = 2;
  void clear_y();
  static const int kYFieldNumber = 2;
  ::google::protobuf::int32 y() const;
  void set_y(::google::protobuf::int32 value);

  // float yaw = 3;
  void clear_yaw();
  static const int kYawFieldNumber = 3;
  float yaw() const;
  void set_yaw(float value);

  // @@protoc_insertion_point(class_scope:kodama.SensorData.Pose2D)
 private:
  class HasBitSetters;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::int32 x_;
  ::google::protobuf::int32 y_;
  float yaw_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_kodama_5fmsg_2eproto;
};
// -------------------------------------------------------------------

class SensorData :
    public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:kodama.SensorData) */ {
 public:
  SensorData();
  virtual ~SensorData();

  SensorData(const SensorData& from);

  inline SensorData& operator=(const SensorData& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  SensorData(SensorData&& from) noexcept
    : SensorData() {
    *this = ::std::move(from);
  }

  inline SensorData& operator=(SensorData&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor() {
    return default_instance().GetDescriptor();
  }
  static const SensorData& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const SensorData* internal_default_instance() {
    return reinterpret_cast<const SensorData*>(
               &_SensorData_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    3;

  void Swap(SensorData* other);
  friend void swap(SensorData& a, SensorData& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline SensorData* New() const final {
    return CreateMaybeMessage<SensorData>(nullptr);
  }

  SensorData* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<SensorData>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const SensorData& from);
  void MergeFrom(const SensorData& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  static const char* _InternalParse(const char* begin, const char* end, void* object, ::google::protobuf::internal::ParseContext* ctx);
  ::google::protobuf::internal::ParseFunc _ParseFunc() const final { return _InternalParse; }
  #else
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(SensorData* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef SensorData_Position2D Position2D;
  typedef SensorData_Pose2D Pose2D;

  // accessors -------------------------------------------------------

  // repeated .kodama.SensorData.Pose2D nearby_robot_poses = 2;
  int nearby_robot_poses_size() const;
  void clear_nearby_robot_poses();
  static const int kNearbyRobotPosesFieldNumber = 2;
  ::kodama::SensorData_Pose2D* mutable_nearby_robot_poses(int index);
  ::google::protobuf::RepeatedPtrField< ::kodama::SensorData_Pose2D >*
      mutable_nearby_robot_poses();
  const ::kodama::SensorData_Pose2D& nearby_robot_poses(int index) const;
  ::kodama::SensorData_Pose2D* add_nearby_robot_poses();
  const ::google::protobuf::RepeatedPtrField< ::kodama::SensorData_Pose2D >&
      nearby_robot_poses() const;

  // repeated .kodama.SensorData.Position2D nearby_target_positions = 3;
  int nearby_target_positions_size() const;
  void clear_nearby_target_positions();
  static const int kNearbyTargetPositionsFieldNumber = 3;
  ::kodama::SensorData_Position2D* mutable_nearby_target_positions(int index);
  ::google::protobuf::RepeatedPtrField< ::kodama::SensorData_Position2D >*
      mutable_nearby_target_positions();
  const ::kodama::SensorData_Position2D& nearby_target_positions(int index) const;
  ::kodama::SensorData_Position2D* add_nearby_target_positions();
  const ::google::protobuf::RepeatedPtrField< ::kodama::SensorData_Position2D >&
      nearby_target_positions() const;

  // .kodama.SensorData.Pose2D pose = 1;
  bool has_pose() const;
  void clear_pose();
  static const int kPoseFieldNumber = 1;
  const ::kodama::SensorData_Pose2D& pose() const;
  ::kodama::SensorData_Pose2D* release_pose();
  ::kodama::SensorData_Pose2D* mutable_pose();
  void set_allocated_pose(::kodama::SensorData_Pose2D* pose);

  // .google.protobuf.Timestamp timestamp = 4;
  bool has_timestamp() const;
  void clear_timestamp();
  static const int kTimestampFieldNumber = 4;
  const ::google::protobuf::Timestamp& timestamp() const;
  ::google::protobuf::Timestamp* release_timestamp();
  ::google::protobuf::Timestamp* mutable_timestamp();
  void set_allocated_timestamp(::google::protobuf::Timestamp* timestamp);

  // @@protoc_insertion_point(class_scope:kodama.SensorData)
 private:
  class HasBitSetters;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedPtrField< ::kodama::SensorData_Pose2D > nearby_robot_poses_;
  ::google::protobuf::RepeatedPtrField< ::kodama::SensorData_Position2D > nearby_target_positions_;
  ::kodama::SensorData_Pose2D* pose_;
  ::google::protobuf::Timestamp* timestamp_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_kodama_5fmsg_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// RequestData

// int32 tag_id = 1;
inline void RequestData::clear_tag_id() {
  tag_id_ = 0;
}
inline ::google::protobuf::int32 RequestData::tag_id() const {
  // @@protoc_insertion_point(field_get:kodama.RequestData.tag_id)
  return tag_id_;
}
inline void RequestData::set_tag_id(::google::protobuf::int32 value) {
  
  tag_id_ = value;
  // @@protoc_insertion_point(field_set:kodama.RequestData.tag_id)
}

// int32 v = 2;
inline void RequestData::clear_v() {
  v_ = 0;
}
inline ::google::protobuf::int32 RequestData::v() const {
  // @@protoc_insertion_point(field_get:kodama.RequestData.v)
  return v_;
}
inline void RequestData::set_v(::google::protobuf::int32 value) {
  
  v_ = value;
  // @@protoc_insertion_point(field_set:kodama.RequestData.v)
}

// int32 w = 3;
inline void RequestData::clear_w() {
  w_ = 0;
}
inline ::google::protobuf::int32 RequestData::w() const {
  // @@protoc_insertion_point(field_get:kodama.RequestData.w)
  return w_;
}
inline void RequestData::set_w(::google::protobuf::int32 value) {
  
  w_ = value;
  // @@protoc_insertion_point(field_set:kodama.RequestData.w)
}

// int32 tau = 4;
inline void RequestData::clear_tau() {
  tau_ = 0;
}
inline ::google::protobuf::int32 RequestData::tau() const {
  // @@protoc_insertion_point(field_get:kodama.RequestData.tau)
  return tau_;
}
inline void RequestData::set_tau(::google::protobuf::int32 value) {
  
  tau_ = value;
  // @@protoc_insertion_point(field_set:kodama.RequestData.tau)
}

// int32 scenario = 6;
inline void RequestData::clear_scenario() {
  scenario_ = 0;
}
inline ::google::protobuf::int32 RequestData::scenario() const {
  // @@protoc_insertion_point(field_get:kodama.RequestData.scenario)
  return scenario_;
}
inline void RequestData::set_scenario(::google::protobuf::int32 value) {
  
  scenario_ = value;
  // @@protoc_insertion_point(field_set:kodama.RequestData.scenario)
}

// int32 targetX = 7;
inline void RequestData::clear_targetx() {
  targetx_ = 0;
}
inline ::google::protobuf::int32 RequestData::targetx() const {
  // @@protoc_insertion_point(field_get:kodama.RequestData.targetX)
  return targetx_;
}
inline void RequestData::set_targetx(::google::protobuf::int32 value) {
  
  targetx_ = value;
  // @@protoc_insertion_point(field_set:kodama.RequestData.targetX)
}

// int32 targetY = 8;
inline void RequestData::clear_targety() {
  targety_ = 0;
}
inline ::google::protobuf::int32 RequestData::targety() const {
  // @@protoc_insertion_point(field_get:kodama.RequestData.targetY)
  return targety_;
}
inline void RequestData::set_targety(::google::protobuf::int32 value) {
  
  targety_ = value;
  // @@protoc_insertion_point(field_set:kodama.RequestData.targetY)
}

// -------------------------------------------------------------------

// SensorData_Position2D

// int32 x = 1;
inline void SensorData_Position2D::clear_x() {
  x_ = 0;
}
inline ::google::protobuf::int32 SensorData_Position2D::x() const {
  // @@protoc_insertion_point(field_get:kodama.SensorData.Position2D.x)
  return x_;
}
inline void SensorData_Position2D::set_x(::google::protobuf::int32 value) {
  
  x_ = value;
  // @@protoc_insertion_point(field_set:kodama.SensorData.Position2D.x)
}

// int32 y = 2;
inline void SensorData_Position2D::clear_y() {
  y_ = 0;
}
inline ::google::protobuf::int32 SensorData_Position2D::y() const {
  // @@protoc_insertion_point(field_get:kodama.SensorData.Position2D.y)
  return y_;
}
inline void SensorData_Position2D::set_y(::google::protobuf::int32 value) {
  
  y_ = value;
  // @@protoc_insertion_point(field_set:kodama.SensorData.Position2D.y)
}

// -------------------------------------------------------------------

// SensorData_Pose2D

// int32 x = 1;
inline void SensorData_Pose2D::clear_x() {
  x_ = 0;
}
inline ::google::protobuf::int32 SensorData_Pose2D::x() const {
  // @@protoc_insertion_point(field_get:kodama.SensorData.Pose2D.x)
  return x_;
}
inline void SensorData_Pose2D::set_x(::google::protobuf::int32 value) {
  
  x_ = value;
  // @@protoc_insertion_point(field_set:kodama.SensorData.Pose2D.x)
}

// int32 y = 2;
inline void SensorData_Pose2D::clear_y() {
  y_ = 0;
}
inline ::google::protobuf::int32 SensorData_Pose2D::y() const {
  // @@protoc_insertion_point(field_get:kodama.SensorData.Pose2D.y)
  return y_;
}
inline void SensorData_Pose2D::set_y(::google::protobuf::int32 value) {
  
  y_ = value;
  // @@protoc_insertion_point(field_set:kodama.SensorData.Pose2D.y)
}

// float yaw = 3;
inline void SensorData_Pose2D::clear_yaw() {
  yaw_ = 0;
}
inline float SensorData_Pose2D::yaw() const {
  // @@protoc_insertion_point(field_get:kodama.SensorData.Pose2D.yaw)
  return yaw_;
}
inline void SensorData_Pose2D::set_yaw(float value) {
  
  yaw_ = value;
  // @@protoc_insertion_point(field_set:kodama.SensorData.Pose2D.yaw)
}

// -------------------------------------------------------------------

// SensorData

// .kodama.SensorData.Pose2D pose = 1;
inline bool SensorData::has_pose() const {
  return this != internal_default_instance() && pose_ != nullptr;
}
inline void SensorData::clear_pose() {
  if (GetArenaNoVirtual() == nullptr && pose_ != nullptr) {
    delete pose_;
  }
  pose_ = nullptr;
}
inline const ::kodama::SensorData_Pose2D& SensorData::pose() const {
  const ::kodama::SensorData_Pose2D* p = pose_;
  // @@protoc_insertion_point(field_get:kodama.SensorData.pose)
  return p != nullptr ? *p : *reinterpret_cast<const ::kodama::SensorData_Pose2D*>(
      &::kodama::_SensorData_Pose2D_default_instance_);
}
inline ::kodama::SensorData_Pose2D* SensorData::release_pose() {
  // @@protoc_insertion_point(field_release:kodama.SensorData.pose)
  
  ::kodama::SensorData_Pose2D* temp = pose_;
  pose_ = nullptr;
  return temp;
}
inline ::kodama::SensorData_Pose2D* SensorData::mutable_pose() {
  
  if (pose_ == nullptr) {
    auto* p = CreateMaybeMessage<::kodama::SensorData_Pose2D>(GetArenaNoVirtual());
    pose_ = p;
  }
  // @@protoc_insertion_point(field_mutable:kodama.SensorData.pose)
  return pose_;
}
inline void SensorData::set_allocated_pose(::kodama::SensorData_Pose2D* pose) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == nullptr) {
    delete pose_;
  }
  if (pose) {
    ::google::protobuf::Arena* submessage_arena = nullptr;
    if (message_arena != submessage_arena) {
      pose = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, pose, submessage_arena);
    }
    
  } else {
    
  }
  pose_ = pose;
  // @@protoc_insertion_point(field_set_allocated:kodama.SensorData.pose)
}

// repeated .kodama.SensorData.Pose2D nearby_robot_poses = 2;
inline int SensorData::nearby_robot_poses_size() const {
  return nearby_robot_poses_.size();
}
inline void SensorData::clear_nearby_robot_poses() {
  nearby_robot_poses_.Clear();
}
inline ::kodama::SensorData_Pose2D* SensorData::mutable_nearby_robot_poses(int index) {
  // @@protoc_insertion_point(field_mutable:kodama.SensorData.nearby_robot_poses)
  return nearby_robot_poses_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::kodama::SensorData_Pose2D >*
SensorData::mutable_nearby_robot_poses() {
  // @@protoc_insertion_point(field_mutable_list:kodama.SensorData.nearby_robot_poses)
  return &nearby_robot_poses_;
}
inline const ::kodama::SensorData_Pose2D& SensorData::nearby_robot_poses(int index) const {
  // @@protoc_insertion_point(field_get:kodama.SensorData.nearby_robot_poses)
  return nearby_robot_poses_.Get(index);
}
inline ::kodama::SensorData_Pose2D* SensorData::add_nearby_robot_poses() {
  // @@protoc_insertion_point(field_add:kodama.SensorData.nearby_robot_poses)
  return nearby_robot_poses_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::kodama::SensorData_Pose2D >&
SensorData::nearby_robot_poses() const {
  // @@protoc_insertion_point(field_list:kodama.SensorData.nearby_robot_poses)
  return nearby_robot_poses_;
}

// repeated .kodama.SensorData.Position2D nearby_target_positions = 3;
inline int SensorData::nearby_target_positions_size() const {
  return nearby_target_positions_.size();
}
inline void SensorData::clear_nearby_target_positions() {
  nearby_target_positions_.Clear();
}
inline ::kodama::SensorData_Position2D* SensorData::mutable_nearby_target_positions(int index) {
  // @@protoc_insertion_point(field_mutable:kodama.SensorData.nearby_target_positions)
  return nearby_target_positions_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::kodama::SensorData_Position2D >*
SensorData::mutable_nearby_target_positions() {
  // @@protoc_insertion_point(field_mutable_list:kodama.SensorData.nearby_target_positions)
  return &nearby_target_positions_;
}
inline const ::kodama::SensorData_Position2D& SensorData::nearby_target_positions(int index) const {
  // @@protoc_insertion_point(field_get:kodama.SensorData.nearby_target_positions)
  return nearby_target_positions_.Get(index);
}
inline ::kodama::SensorData_Position2D* SensorData::add_nearby_target_positions() {
  // @@protoc_insertion_point(field_add:kodama.SensorData.nearby_target_positions)
  return nearby_target_positions_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::kodama::SensorData_Position2D >&
SensorData::nearby_target_positions() const {
  // @@protoc_insertion_point(field_list:kodama.SensorData.nearby_target_positions)
  return nearby_target_positions_;
}

// .google.protobuf.Timestamp timestamp = 4;
inline bool SensorData::has_timestamp() const {
  return this != internal_default_instance() && timestamp_ != nullptr;
}
inline const ::google::protobuf::Timestamp& SensorData::timestamp() const {
  const ::google::protobuf::Timestamp* p = timestamp_;
  // @@protoc_insertion_point(field_get:kodama.SensorData.timestamp)
  return p != nullptr ? *p : *reinterpret_cast<const ::google::protobuf::Timestamp*>(
      &::google::protobuf::_Timestamp_default_instance_);
}
inline ::google::protobuf::Timestamp* SensorData::release_timestamp() {
  // @@protoc_insertion_point(field_release:kodama.SensorData.timestamp)
  
  ::google::protobuf::Timestamp* temp = timestamp_;
  timestamp_ = nullptr;
  return temp;
}
inline ::google::protobuf::Timestamp* SensorData::mutable_timestamp() {
  
  if (timestamp_ == nullptr) {
    auto* p = CreateMaybeMessage<::google::protobuf::Timestamp>(GetArenaNoVirtual());
    timestamp_ = p;
  }
  // @@protoc_insertion_point(field_mutable:kodama.SensorData.timestamp)
  return timestamp_;
}
inline void SensorData::set_allocated_timestamp(::google::protobuf::Timestamp* timestamp) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(timestamp_);
  }
  if (timestamp) {
    ::google::protobuf::Arena* submessage_arena =
      reinterpret_cast<::google::protobuf::MessageLite*>(timestamp)->GetArena();
    if (message_arena != submessage_arena) {
      timestamp = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, timestamp, submessage_arena);
    }
    
  } else {
    
  }
  timestamp_ = timestamp;
  // @@protoc_insertion_point(field_set_allocated:kodama.SensorData.timestamp)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace kodama

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // PROTOBUF_INCLUDED_kodama_5fmsg_2eproto
