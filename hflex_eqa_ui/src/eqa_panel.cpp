#include "hflex_eqa_ui/eqa_panel.hpp"

#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QMetaObject>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/config.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

namespace hflex_eqa_ui {

namespace {

QString toQString(const std::string& value) { return QString::fromStdString(value); }

constexpr auto kReadyTopic = "/eqa/ready";
constexpr auto kQuestionTopic = "/eqa_question";
constexpr auto kChoicesTopic = "/eqa_choices";
constexpr auto kTriggerStepService = "/hflex_eqa/search_manager/trigger_step";
constexpr auto kTriggerStopService = "/hflex_eqa/search_manager/trigger_stop";
constexpr auto kTriggerNextObjectService =
    "/hflex_eqa/search_manager/trigger_next_object";

QStringList parseChoices(const QString& choices_text) {
  QStringList choices;
  for (const auto& line : choices_text.split('\n', Qt::SkipEmptyParts)) {
    const auto choice = line.trimmed();
    if (!choice.isEmpty()) {
      choices.push_back(choice);
    }
  }

  return choices;
}

std::string toJsonArray(const QStringList& choices) {
  QJsonArray choices_json;
  for (const auto& choice : choices) {
    choices_json.append(choice);
  }

  return QJsonDocument(choices_json).toJson(QJsonDocument::Compact).toStdString();
}

}  // namespace

EQAPanel::EQAPanel(QWidget* parent) : rviz_common::Panel(parent) {
  auto* main_layout = new QVBoxLayout;

  start_button_ = new QPushButton("Start EQA");
  main_layout->addWidget(start_button_);

  stop_button_ = new QPushButton("Stop");
  main_layout->addWidget(stop_button_);

  trigger_step_button_ = new QPushButton("Trigger EQA Step");
  main_layout->addWidget(trigger_step_button_);

  trigger_next_object_button_ = new QPushButton("Go To Next Object");
  main_layout->addWidget(trigger_next_object_button_);

  auto* question_layout = new QHBoxLayout;
  question_edit_ = new QLineEdit;
  question_edit_->setPlaceholderText("Question");
  send_button_ = new QPushButton("Send");
  question_layout->addWidget(new QLabel("Question"));
  question_layout->addWidget(question_edit_, 1);
  question_layout->addWidget(send_button_);
  main_layout->addLayout(question_layout);

  auto* choices_layout = new QVBoxLayout;
  choices_edit_ = new QPlainTextEdit;
  choices_edit_->setPlaceholderText("One choice per line");
  choices_edit_->setMaximumHeight(90);
  choices_layout->addWidget(new QLabel("Choices"));
  choices_layout->addWidget(choices_edit_);
  main_layout->addLayout(choices_layout);

  status_edit_ = new QLineEdit;
  status_edit_->setReadOnly(true);
  main_layout->addWidget(status_edit_);

  setLayout(main_layout);

  connect(start_button_.data(), &QPushButton::clicked, this, &EQAPanel::publishReady);
  connect(stop_button_.data(), &QPushButton::clicked, this, &EQAPanel::triggerStop);
  connect(trigger_step_button_.data(),
          &QPushButton::clicked,
          this,
          &EQAPanel::triggerEQAStep);
  connect(trigger_next_object_button_.data(),
          &QPushButton::clicked,
          this,
          &EQAPanel::triggerNextObject);
  connect(send_button_.data(), &QPushButton::clicked, this, &EQAPanel::publishQuestion);
  connect(question_edit_.data(),
          &QLineEdit::returnPressed,
          this,
          &EQAPanel::publishQuestion);

  start_button_->setEnabled(false);
  stop_button_->setEnabled(false);
  trigger_step_button_->setEnabled(false);
  trigger_next_object_button_->setEnabled(false);
  send_button_->setEnabled(false);
}

void EQAPanel::onInitialize() {
  const auto node_abstraction = getDisplayContext()->getRosNodeAbstraction().lock();
  if (!node_abstraction) {
    setStatus("Failed to access RViz ROS node");
    return;
  }

  node_ = node_abstraction->get_raw_node();
  createPublishers();
  createClients();
}

void EQAPanel::save(rviz_common::Config config) const {
  rviz_common::Panel::save(config);
}

void EQAPanel::load(const rviz_common::Config& config) {
  rviz_common::Panel::load(config);
}

void EQAPanel::publishReady() {
  if (!ready_pub_) {
    return;
  }

  std_msgs::msg::Bool msg;
  msg.data = true;
  ready_pub_->publish(msg);
  setStatus("Published ready=true to " + QString(kReadyTopic));
}

void EQAPanel::publishQuestion() {
  if (!question_pub_ || !choices_pub_) {
    return;
  }

  const auto question = question_edit_->text().trimmed();
  if (question.isEmpty()) {
    return;
  }

  const auto choices = parseChoices(choices_edit_->toPlainText());

  std_msgs::msg::String choices_msg;
  choices_msg.data = toJsonArray(choices);
  choices_pub_->publish(choices_msg);

  std_msgs::msg::String msg;
  msg.data = question.toStdString();
  question_pub_->publish(msg);
  setStatus("Published question and " + QString::number(choices.size()) + " choices");
}

void EQAPanel::triggerEQAStep() {
  if (!trigger_step_client_) {
    return;
  }

  if (!trigger_step_client_->service_is_ready()) {
    setStatus("Service not available: " + QString(kTriggerStepService));
    return;
  }

  trigger_step_button_->setEnabled(false);
  setStatus("Calling " + QString(kTriggerStepService));

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  QPointer<EQAPanel> panel(this);
  trigger_step_client_->async_send_request(
      request, [panel](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
        if (!panel) {
          return;
        }

        const auto response = future.get();
        const QString status =
            response->success ? "Triggered EQA step" : "Failed to trigger EQA step";
        const QString message = toQString(response->message);
        QMetaObject::invokeMethod(
            panel,
            [panel, status, message]() {
              if (!panel) {
                return;
              }

              panel->trigger_step_button_->setEnabled(true);
              panel->setStatus(message.isEmpty() ? status : status + ": " + message);
            },
            Qt::QueuedConnection);
      });
}

void EQAPanel::triggerStop() {
  if (!trigger_stop_client_) {
    return;
  }

  if (!trigger_stop_client_->service_is_ready()) {
    setStatus("Service not available: " + QString(kTriggerStopService));
    return;
  }

  stop_button_->setEnabled(false);
  setStatus("Calling " + QString(kTriggerStopService));

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  QPointer<EQAPanel> panel(this);
  trigger_stop_client_->async_send_request(
      request, [panel](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
        if (!panel) {
          return;
        }

        const auto response = future.get();
        const QString status =
            response->success ? "Triggered stop" : "Failed to trigger stop";
        const QString message = toQString(response->message);
        QMetaObject::invokeMethod(
            panel,
            [panel, status, message]() {
              if (!panel) {
                return;
              }

              panel->stop_button_->setEnabled(true);
              panel->setStatus(message.isEmpty() ? status : status + ": " + message);
            },
            Qt::QueuedConnection);
      });
}

void EQAPanel::triggerNextObject() {
  if (!trigger_next_object_client_) {
    return;
  }

  if (!trigger_next_object_client_->service_is_ready()) {
    setStatus("Service not available: " + QString(kTriggerNextObjectService));
    return;
  }

  trigger_next_object_button_->setEnabled(false);
  setStatus("Calling " + QString(kTriggerNextObjectService));

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  QPointer<EQAPanel> panel(this);
  trigger_next_object_client_->async_send_request(
      request, [panel](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
        if (!panel) {
          return;
        }

        const auto response = future.get();
        const QString status = response->success ? "Triggered go to next object"
                                                 : "Failed to trigger next object";
        const QString message = toQString(response->message);
        QMetaObject::invokeMethod(
            panel,
            [panel, status, message]() {
              if (!panel) {
                return;
              }

              panel->trigger_next_object_button_->setEnabled(true);
              panel->setStatus(message.isEmpty() ? status : status + ": " + message);
            },
            Qt::QueuedConnection);
      });
}

void EQAPanel::createPublishers() {
  if (!node_) {
    return;
  }

  try {
    ready_pub_ = node_->create_publisher<std_msgs::msg::Bool>(
        kReadyTopic, rclcpp::QoS(1).transient_local());
    question_pub_ = node_->create_publisher<std_msgs::msg::String>(kQuestionTopic, 1);
    choices_pub_ = node_->create_publisher<std_msgs::msg::String>(kChoicesTopic, 1);
    start_button_->setEnabled(true);
    send_button_->setEnabled(true);
  } catch (const std::exception& e) {
    start_button_->setEnabled(false);
    send_button_->setEnabled(false);
    RCLCPP_ERROR(
        node_->get_logger(), "Failed to create EQA UI publishers: %s", e.what());
  }
}

void EQAPanel::createClients() {
  if (!node_) {
    return;
  }

  try {
    trigger_step_client_ =
        node_->create_client<std_srvs::srv::Trigger>(kTriggerStepService);
    trigger_stop_client_ =
        node_->create_client<std_srvs::srv::Trigger>(kTriggerStopService);
    trigger_next_object_client_ =
        node_->create_client<std_srvs::srv::Trigger>(kTriggerNextObjectService);
    trigger_step_button_->setEnabled(true);
    stop_button_->setEnabled(true);
    trigger_next_object_button_->setEnabled(true);
  } catch (const std::exception& e) {
    trigger_step_button_->setEnabled(false);
    stop_button_->setEnabled(false);
    trigger_next_object_button_->setEnabled(false);
    RCLCPP_ERROR(
        node_->get_logger(), "Failed to create EQA UI service clients: %s", e.what());
  }
}

void EQAPanel::setStatus(const QString& status) {
  if (status_edit_) {
    status_edit_->setText(status);
  }
}

}  // namespace hflex_eqa_ui

PLUGINLIB_EXPORT_CLASS(hflex_eqa_ui::EQAPanel, rviz_common::Panel)
