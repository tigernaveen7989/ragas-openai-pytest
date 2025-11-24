import os
import re
import smtplib
from datetime import datetime
from collections import defaultdict

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


class PytestEmailReporter:
    """
    Pytest email reporter that aggregates results by feature (Class::method),
    groups parameterized runs under the same feature, and prepares a summary
    table that contains: Feature | Pass | Fail | Skip | Total | Pass Rate (%)
    """

    def __init__(self, logger):
        self.logger = logger or DummyLogger()

        self.start_time = None
        self.end_time = None

        # overall counters (incremented per test execution)
        self.stats = {"passed": 0, "failed": 0, "skipped": 0}
        self.failure_reasons = {}    # reason -> count
        # list of individual execution records (used for aggregation later)
        # each item: {"file":..., "class":..., "test":..., "status":..., "reason":...}
        self.test_executions = []

        # path to pytest-html detailed report (attached if present)
        self.report_path = "testreports/report.html"

        # project name for the summary table
        self.project_name = os.getenv("PROJECT_NAME", "Pytest-OpenAI-Ragas")

    # -------------------------------------------------------------------------
    def classify_error(self, msg: str):
        if not msg:
            return "UnknownError"
        text = msg.lower()
        if "assert" in text:
            return "AssertionError"
        if "timeout" in text:
            return "TimeoutError"
        if "not found" in text or "locator" in text:
            return "LocatorError"
        if "invalid" in text or "data" in text:
            return "TestDataError"
        if "connection" in text:
            return "ConnectionError"
        if "permission" in text:
            return "PermissionError"
        return "UnknownError"

    # -------------------------------------------------------------------------
    def _normalize_feature(self, nodeid: str):
        """
        Build feature identifier in the format Class::method (Option A).
        Nodeid examples:
          - tests/test_api.py::TestRestAssured::test_aspect_critic[get_multiturn_data0]
          - tests/test_api.py::TestRestAssured::test_aspect_critic
          - tests/test_api.py::test_standalone_function[param]
        Behavior:
          - If a class is present, feature -> Class::method
          - If no class, fallback to file::method
          - Remove parameter suffix like [param...]
        """
        # first split by ::
        parts = nodeid.split("::")

        # remove trailing paramization [..] from last part
        def strip_params(s):
            return re.sub(r"\[.*\]$", "", s)

        if len(parts) >= 3:
            # typical: file :: class :: method[param]
            class_name = strip_params(parts[1])
            method = strip_params(parts[2])
            feature = f"{class_name}::{method}"
            return feature
        elif len(parts) == 2:
            # could be file :: function_or_class_method
            second = strip_params(parts[1])
            # guess if second looks like Class.method? but pytest uses :: so likely function => use file::func
            file_part = os.path.basename(parts[0])
            feature = f"{file_part}::{second}"
            return feature
        else:
            # fallback: use full nodeid without params
            return strip_params(nodeid)

    # -------------------------------------------------------------------------
    def pytest_runtest_logreport(self, report):
        """
        Called by pytest for each test phase report. We only care about the 'call' phase.
        """
        if report.when != "call":
            return

        outcome = report.outcome  # 'passed' | 'failed' | 'skipped'
        try:
            self.stats[outcome] += 1
        except KeyError:
            # unexpected status; count as unknown under 'skipped' just to be safe
            self.stats.setdefault(outcome, 0)
            self.stats[outcome] += 1

        nodeid = getattr(report, "nodeid", "")
        feature = self._normalize_feature(nodeid)

        failure_type = None
        if outcome == "failed":
            failure_type = self.classify_error(str(report.longrepr))
            self.failure_reasons[failure_type] = self.failure_reasons.get(failure_type, 0) + 1
            self.logger.error(f"❌ Test failed → {failure_type}")

        # Save the raw execution (we will aggregate by feature at the end)
        self.test_executions.append({
            "nodeid": nodeid,
            "feature": feature,
            "status": outcome,
            "reason": failure_type or "-"
        })

    # -------------------------------------------------------------------------
    def session_start(self):
        self.start_time = datetime.now()
        self.logger.info(f"➡ Pytest session started at {self.start_time}")

    def session_end(self):
        self.end_time = datetime.now()
        self.logger.info(f"⬅ Pytest session finished at {self.end_time}")

    # -------------------------------------------------------------------------
    def _aggregate_by_feature(self):
        """
        Aggregate self.test_executions into a summary per feature:
        feature_summary = {
            feature_key: {"passed": n, "failed": n, "skipped": n, "total": n, "pass_rate": x.xx}
        }
        """
        feat_counts = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0, "total": 0})
        for exec_rec in self.test_executions:
            f = exec_rec["feature"]
            st = exec_rec["status"]
            feat_counts[f][st] = feat_counts[f].get(st, 0) + 1
            feat_counts[f]["total"] += 1

        # compute pass rate per feature
        feature_summary = {}
        for f, counts in feat_counts.items():
            total = counts.get("total", 0)
            passed = counts.get("passed", 0)
            pass_rate = round((passed / total * 100), 2) if total else 0.0
            feature_summary[f] = {
                "passed": passed,
                "failed": counts.get("failed", 0),
                "skipped": counts.get("skipped", 0),
                "total": total,
                "pass_rate": pass_rate
            }
        return feature_summary

    # -------------------------------------------------------------------------
    def generate_html(self):
        """
        Generate email HTML. The Feature Summary table will show:
          Feature | Pass | Fail | Skip | Total | Pass Rate (%)
        Only aggregated feature rows (Class::method) are shown.
        """
        passed = self.stats.get("passed", 0)
        failed = self.stats.get("failed", 0)
        skipped = self.stats.get("skipped", 0)
        total = passed + failed + skipped
        overall_pass_rate = round((passed / total * 100), 2) if total else 0.0

        # pie chart for quick visual
        pass_pct = (passed / total * 100) if total else 0
        fail_pct = (failed / total * 100) if total else 0

        pie_chart = f"""
            background: conic-gradient(
                #2e7d32 0% {pass_pct}%,
                #c62828 {pass_pct}% {pass_pct + fail_pct}%,
                #ef6c00 {pass_pct + fail_pct}% 100%
            );
        """

        # aggregated feature summary
        feature_summary = self._aggregate_by_feature()

        # build feature summary rows sorted by feature name for stability
        feature_rows = ""
        for feature in sorted(feature_summary.keys()):
            s = feature_summary[feature]
            feature_rows += f"""
            <tr>
                <td style="text-align:left;padding-left:10px;">{feature}</td>
                <td>{s['passed']}</td>
                <td>{s['failed']}</td>
                <td>{s['skipped']}</td>
                <td>{s['total']}</td>
                <td>{s['pass_rate']}%</td>
            </tr>
            """

        # failure breakdown rows
        failure_rows = ""
        if self.failure_reasons:
            for reason, count in sorted(self.failure_reasons.items(), key=lambda x: -x[1]):
                failure_rows += f"<tr><td>{reason}</td><td><b>{count}</b></td></tr>"
        else:
            failure_rows = "<tr><td colspan='2'>No Failures 🎉</td></tr>"

        # HTML body
        return f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, Helvetica, sans-serif;
                    padding: 18px;
                }}
                h2 {{ text-align: center; margin-bottom: 6px; }}
                .pie {{
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    margin: 10px auto 20px;
                    {pie_chart}
                }}
                table {{
                    width: 95%;
                    margin: 10px auto;
                    border-collapse: collapse;
                    font-size: 13px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: center;
                }}
                th {{
                    background-color: #222;
                    color: #fff;
                    font-weight: 600;
                }}
                tr:nth-child(even) {{ background: #fafafa; }}
                .small-table {{ width: 50%; margin: 10px auto; }}
                .left-col {{ text-align: left; padding-left: 12px; }}
            </style>
        </head>
        <body>
            <h2>📊 Pytest Execution Summary</h2>
            <div class="pie"></div>

            <table>
                <tr>
                    <th>Project Name</th>
                    <th>Total Tests</th>
                    <th>Passed</th>
                    <th>Failed</th>
                    <th>Skipped</th>
                    <th>Pass Rate (%)</th>
                </tr>
                <tr>
                    <td>{self.project_name}</td>
                    <td>{total}</td>
                    <td>{passed}</td>
                    <td>{failed}</td>
                    <td>{skipped}</td>
                    <td>{overall_pass_rate}%</td>
                </tr>
            </table>

            <h3 style="text-align:center;">Feature Summary (Class :: Test Method)</h3>
            <table>
                <tr>
                    <th style="text-align:left;padding-left:10px;">Feature</th>
                    <th>Pass</th>
                    <th>Fail</th>
                    <th>Skip</th>
                    <th>Total</th>
                    <th>Pass Rate</th>
                </tr>
                {feature_rows or '<tr><td colspan="6">No tests executed</td></tr>'}
            </table>

            <h3 style="text-align:center;">Failure Breakdown</h3>
            <table class="small-table">
                <tr><th>Failure Type</th><th>Count</th></tr>
                {failure_rows}
            </table>

            <h3 style="text-align:center;">Execution Time</h3>
            <table class="small-table">
                <tr><td>Start Time</td><td>{self.start_time}</td></tr>
                <tr><td>End Time</td><td>{self.end_time}</td></tr>
            </table>
        </body>
        </html>
        """

    # -------------------------------------------------------------------------
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

        # Build HTML
        html = self.generate_html()

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(receivers)
        msg["Subject"] = "📢 Pytest Execution Summary"
        msg.attach(MIMEText(html, "html"))

        # Attach pytest-html report if present
        if os.path.exists(self.report_path):
            try:
                with open(self.report_path, "rb") as f:
                    part = MIMEApplication(f.read(), Name=os.path.basename(self.report_path))
                    part["Content-Disposition"] = f'attachment; filename="{os.path.basename(self.report_path)}"'
                    msg.attach(part)
            except Exception as e:
                # log but continue sending email with aggregated HTML
                self.logger.warning(f"⚠ Could not attach pytest-html report: {e}")

        # send email
        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                try:
                    server.starttls()
                except Exception:
                    # some SMTP servers don't support STARTTLS on the port used
                    self.logger.warning("⚠ TLS not supported by SMTP server or not needed")
                server.sendmail(sender, receivers, msg.as_string())
                self.logger.info("📧 Email report sent successfully!")
        except Exception as e:
            self.logger.error(f"❌ Failed to send email: {e}")

# -------------------------------------------------------------------------
class DummyLogger:
    def info(self, msg):
        print("[INFO]", msg)

    def warning(self, msg):
        print("[WARN]", msg)

    def error(self, msg):
        print("[ERROR]", msg)
