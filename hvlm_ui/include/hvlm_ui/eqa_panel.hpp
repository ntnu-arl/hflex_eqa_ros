#pragma once

#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPointer>
#include <QPushButton>
#include <QString>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/config.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

class QVBoxLayout;

namespace hvlm_ui {

class EQAPanel : public rviz_common::Panel {
  Q_OBJECT

 public:
  explicit EQAPanel(QWidget* parent = nullptr);

  void onInitialize() override;
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config& config) override;

 private Q_SLOTS:
  void publishReady();
  void publishQuestion();
  void triggerEQAStep();
  void triggerStop();
  void triggerNextObject();

 private:
  void createPublishers();
  void createClients();
  void setStatus(const QString& status);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ready_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr question_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr choices_pub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr trigger_step_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr trigger_stop_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr trigger_next_object_client_;

  QPointer<QPushButton> start_button_;
  QPointer<QPushButton> stop_button_;
  QPointer<QPushButton> trigger_step_button_;
  QPointer<QPushButton> trigger_next_object_button_;
  QPointer<QLineEdit> question_edit_;
  QPointer<QPlainTextEdit> choices_edit_;
  QPointer<QPushButton> send_button_;
  QPointer<QLineEdit> status_edit_;
};

}  // namespace hvlm_ui
