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
    if not EMAIL_ENABLED:
        logger.info("Email alerts disabled")
        return

    msg = MIMEMultipart("related")
    msg["Subject"] = "⚠️ Suspicious Transaction Alert"
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_RECEIVER_EMAIL

    # ---------------------------
    # Human explanation list
    # ---------------------------
    human_html = "".join(
        f"<li>{e}</li>" for e in human_explanations
    ) or "<li>Unusual transaction behavior detected</li>"

    # ---------------------------
    # Safe transaction summary
    # ---------------------------
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

    # ---------------------------
    # Technical SHAP section
    # ---------------------------
    technical_html = "".join(
        f"<li><b>{r['feature']}</b>: {r['impact']:.4f}</li>"
        for r in shap_reasons
    )

    # ---------------------------
    # Email HTML
    # ---------------------------
    html = f"""
    <html>
      <body style="font-family:Arial, sans-serif; color:#333;">

        <h2 style="color:#d32f2f;">⚠️ Suspicious Transaction Alert</h2>

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

        <div style="background:#fff3e0; padding:12px; border-left:4px solid #ff9800; margin:16px 0;">
          <b>What should you do?</b>
          <ul>
            <li>If you recognize this transaction, no action is required.</li>
            <li>If you do <b>not</b> recognize it, please contact support immediately.</li>
          </ul>
        </div>

        <div style="background:#f5f5f5; padding:12px; border-left:4px solid #1976d2; margin-bottom:16px;">
          <b>Need help?</b>
          <p style="margin:6px 0;">
            If you believe this transaction is unauthorized, please contact our support team.
          </p>
          <ul>
            <li>Email: <a href="mailto:support@fraudguard.ai">support@fraudguard.ai</a></li>
            <li>Support Hours: 24×7</li>
          </ul>
        </div>

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
          FraudGuard AI • Automated Security Alert<br/>
          We will never ask for your password or OTP via email.
        </small>

      </body>
    </html>
    """

    msg.attach(MIMEText(html, "html"))

    # ---------------------------
    # Attach SHAP image
    # ---------------------------
    if shap_image:
        try:
            img_bytes = base64.b64decode(shap_image)
            image = MIMEImage(img_bytes, _subtype="png")
            image.add_header("Content-ID", "<shap_chart>")
            image.add_header("Content-Disposition", "inline", filename="shap.png")
            msg.attach(image)
        except Exception:
            logger.exception("Failed to attach SHAP image")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)

    logger.info("Fraud email sent successfully")
