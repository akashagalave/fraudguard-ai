import smtplib
import logging
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

from .config import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    ALERT_RECEIVER_EMAIL,
    EMAIL_ENABLED
)

logger = logging.getLogger("email")


def send_fraud_email(
    transaction_id: str,
    features: dict,
    fraud_probability: float,
    risk_scores: int,
    shap_image: str | None,
    shap_reasons: list,
    human_explanations: list
):
    """
    Best-effort email alert.
    MUST NEVER crash inference.
    """

    if not EMAIL_ENABLED:
        logger.info("Email alerts disabled")
        return

    try:
        msg = MIMEMultipart("related")
        msg["Subject"] = " Suspicious Transaction Alert"
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_RECEIVER_EMAIL


        human_html = "".join(
            f"<li>{e}</li>" for e in human_explanations
        ) or "<li>Unusual transaction behavior detected</li>"

        hour = features.get("hour")
        hour_display = f"Hour {hour}" if hour is not None else "N/A"

        txn_summary = f"""
        <table cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
          <tr><td><b>Amount</b></td><td>₹ {features.get('amt')}</td></tr>
          <tr><td><b>Merchant</b></td><td>{features.get('merchant')}</td></tr>
          <tr><td><b>Category</b></td><td>{features.get('category')}</td></tr>
          <tr><td><b>Time</b></td><td>{hour_display}</td></tr>
          <tr><td><b>Reference ID</b></td><td>{transaction_id}</td></tr>
        </table>
        """

        technical_html = "".join(
            f"<li><b>{r['feature']}</b>: {r['impact']:.4f}</li>"
            for r in shap_reasons
        )

        html = f"""
        <html>
          <body style="font-family:Arial, sans-serif; color:#333;">
            <h2 style="color:#d32f2f;"> Suspicious Transaction Alert</h2>

            <p>
              We detected a transaction that appears <b>unusual</b> compared to
              your normal activity.
            </p>

            <p>
              <b>Risk Score:</b> {risk_scores}/100<br/>
              <b>Fraud Probability:</b> {fraud_probability:.6f}
            </p>

            <h3>Why we flagged this transaction</h3>
            <ul>{human_html}</ul>

            <h3>Transaction Summary</h3>
            {txn_summary}
        """

        if shap_image:
            html += """
            <h3>Detailed Explanation (Advanced)</h3>
            <img src="cid:shap_chart" style="max-width:600px; border:1px solid #ddd;"/>
            """

        html += f"""
            <h4>Technical Factors (for analysts)</h4>
            <ul>{technical_html}</ul>

            <hr/>
            <small style="color:#666;">
              FraudGuard AI • Automated Security Alert
            </small>
          </body>
        </html>
        """

        msg.attach(MIMEText(html, "html"))

        if shap_image:
            try:
                img_bytes = base64.b64decode(shap_image)
                image = MIMEImage(img_bytes, _subtype="png")
                image.add_header("Content-ID", "<shap_chart>")
                image.add_header("Content-Disposition", "inline", filename="shap.png")
                msg.attach(image)
            except Exception:
                logger.exception("Failed to attach SHAP image")

        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()

        logger.info("Fraud email sent successfully")

    except Exception as e:

        logger.error("Fraud email failed, continuing inference", exc_info=e)
        return
