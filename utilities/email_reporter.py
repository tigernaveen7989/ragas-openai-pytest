import os
import smtplib
from datetime import datetime

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


class PytestEmailReporter:

    # ------------------------------------------------------------------------------------
    def __init__(self, logger):
        self.logger = logger or DummyLogger()

        self.start_time = None
        self.end_time = None

        self.stats = {"passed": 0, "failed": 0, "skipped": 0}
        self.failure_reasons = {}            # NEW: Count failures by type
        self.test_details = []               # NEW: List of all test entries
        self.report_path = "testreports/report.html"

    # ------------------------------------------------------------------------------------
    def classify_error(self, msg: str):
        if not msg:
            return "UnknownError"

        msg = msg.lower()

        if "assert" in msg:
            return "AssertionError"
        if "timeout" in msg:
            return "TimeoutError"
        if "not found" in msg or "locator" in msg:
            return "LocatorError"
        if "invalid" in msg or "data" in msg:
            return "TestDataError"
        if "connection" in msg:
            return "ConnectionError"
        if "permission" in msg:
            return "PermissionError"

        return "UnknownError"

    # ------------------------------------------------------------------------------------
    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return

        feature_name = report.nodeid
        outcome = report.outcome

        # Count stats
        self.stats[outcome] += 1

        failure_type = None
        if outcome == "failed":
            failure_type = self.classify_error(str(report.longrepr))
            self.failure_reasons[failure_type] = (
                self.failure_reasons.get(failure_type, 0) + 1
            )
            self.logger.error(f"❌ Test failed → {failure_type}")

        # Store row
        self.test_details.append(
            {
                "feature": feature_name,
                "status": outcome,
                "reason": failure_type or "-",
            }
        )

    # ------------------------------------------------------------------------------------
    def session_start(self):
        self.start_time = datetime.now()
        self.logger.info(f"➡ Pytest session started at {self.start_time}")

    def session_end(self):
        self.end_time = datetime.now()
        self.logger.info(f"⬅ Pytest session finished at {self.end_time}")

    # ------------------------------------------------------------------------------------
    def generate_html(self):
        passed = self.stats["passed"]
        failed = self.stats["failed"]
        skipped = self.stats["skipped"]
        total = passed + failed + skipped

        pass_pct = (passed / total * 100) if total else 0
        fail_pct = (failed / total * 100) if total else 0

        pie_chart = f"""
            background: conic-gradient(
                #2e7d32 0% {pass_pct}%,
                #c62828 {pass_pct}% {pass_pct + fail_pct}%,
                #ef6c00 {pass_pct + fail_pct}% 100%
            );
        """

        # Failure breakdown rows
        failure_rows = "".join(
            f"<tr><td>{reason}</td><td><b>{count}</b></td></tr>"
            for reason, count in self.failure_reasons.items()
        ) or "<tr><td colspan='2'>No Failures 🎉</td></tr>"

        # Feature-wise rows
        feature_rows = "".join(
            f"""
            <tr>
                <td>{item['feature']}</td>
                <td>{item['status'].upper()}</td>
                <td>{item['reason']}</td>
            </tr>
            """
            for item in self.test_details
        )

        return f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial;
                    padding: 20px;
                }}
                h2 {{
                    text-align: center;
                }}
                .pie {{
                    width: 140px;
                    height: 140px;
                    border-radius: 50%;
                    margin: 20px auto;
                    {pie_chart}
                }}
                table {{
                    width: 90%;
                    margin: auto;
                    border-collapse: collapse;
                    font-size: 14px;
                }}
                th, td {{
                    border: 1px solid #ccc;
                    padding: 8px;
                    text-align: center;
                }}
                th {{
                    background-color: #222;
                    color: #fff;
                }}
            </style>
        </head>

        <body>
            <h2>📊 Pytest Execution Summary</h2>
            <div class="pie"></div>

            <table>
                <tr>
                    <th>Total</th><th>Passed</th><th>Failed</th><th>Skipped</th>
                </tr>
                <tr>
                    <td>{total}</td><td>{passed}</td><td>{failed}</td><td>{skipped}</td>
                </tr>
            </table>

            <h3 style='text-align:center;'>Feature-wise Results</h3>
            <table>
                <tr>
                    <th>Feature Name</th>
                    <th>Status</th>
                    <th>Failure Type</th>
                </tr>
                {feature_rows}
            </table>

            <h3 style='text-align:center;'>Failure Breakdown</h3>
            <table style="width:50%;">
                <tr><th>Failure Type</th><th>Count</th></tr>
                {failure_rows}
            </table>

            <h3 style='text-align:center;'>Execution Time</h3>
            <table style="width:50%;">
                <tr><td>Start Time</td><td>{self.start_time}</td></tr>
                <tr><td>End Time</td><td>{self.end_time}</td></tr>
            </table>
        </body>
        </html>
        """

    # ------------------------------------------------------------------------------------
    def send_email(self):
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", "25"))
        sender = os.getenv("EMAIL_SENDER")

        receivers = [
            r.strip()
            for r in os.getenv("EMAIL_RECEIVER_LIST", "").split(",")
            if r.strip()
        ]

        if not smtp_host or not sender or not receivers:
            self.logger.error("❌ SMTP configuration missing in .env")
            return

        html = self.generate_html()

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(receivers)
        msg["Subject"] = "📢 Pytest Execution Summary"
        msg.attach(MIMEText(html, "html"))

        # Attach pytest HTML report
        if os.path.exists(self.report_path):
            with open(self.report_path, "rb") as f:
                part = MIMEApplication(f.read(), Name="pytest_report.html")
                part["Content-Disposition"] = (
                    'attachment; filename="pytest_report.html"'
                )
                msg.attach(part)

        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                try:
                    server.starttls()
                except Exception:
                    self.logger.warning("⚠ TLS not supported")

                server.sendmail(sender, receivers, msg.as_string())
                self.logger.info("📧 Email report sent successfully!")

        except Exception as e:
            self.logger.error(f"❌ Failed to send email: {e}")


class DummyLogger:
    def info(self, msg):
        print("[INFO]", msg)

    def warning(self, msg):
        print("[WARN]", msg)

    def error(self, msg):
        print("[ERROR]", msg)
