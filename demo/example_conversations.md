# Example Conversation Logs

These demonstrate all key workflows the AutoStream agent handles.

---

## 1. Greeting Flow

```
User: Hi there!
Agent: Hello! 👋 I'm the AutoStream AI Assistant, powered by Gemini 3.1 Flash-Lite.
       I can help you explore our pricing plans, compare features, or guide you
       through the sign-up process. What can I do for you today?

[State] intent=GREETING | stage=greeting | qualified=No
```

## 2. Product Query (RAG-Grounded)

```
User: What are your pricing plans?
Agent: AutoStream offers two plans:

       **Basic Plan** — $29/month
       - 10 videos per month
       - 720p resolution
       - Email support (business hours)
       - 10 GB storage

       **Pro Plan** — $79/month
       - Unlimited videos
       - 4K resolution
       - AI captions (multi-language)
       - 24/7 priority support
       - Unlimited storage
       - Custom branding & team collaboration (up to 5 seats)

       Would you like a detailed comparison or have questions about a specific plan?

[State] intent=PRODUCT_QUERY | stage=inquiry | qualified=No
[RAG] Retrieved chunks: "Pricing — Basic Plan", "Pricing — Pro Plan"
```

## 3. Refund Policy Query

```
User: What's your refund policy?
Agent: Here's our refund policy:

       - **Full refund** available within the first **7 days** of subscription
       - No refunds after 7 days from purchase
       - Requests must be submitted via the support portal

       Would you like to know about our cancellation policy as well?

[State] intent=PRODUCT_QUERY | stage=inquiry | qualified=No
[RAG] Retrieved chunk: "Refund Policy"
```

## 4. High Intent → Lead Collection Flow

```
User: I'd like to sign up for the Pro plan!
Agent: I'd love to get you started with AutoStream Pro!
       Could you please share your **full name**?

[State] intent=HIGH_INTENT | stage=lead_collect | missing=[name, email, platform]

User: Adarsh Kumar
Agent: Thanks! Now, could you please provide your **professional email address**
       so our team can set up your account?

[State] intent=HIGH_INTENT | stage=lead_collect | lead_name=Adarsh Kumar | missing=[email, platform]

User: adarsh@example.com
Agent: Almost there! Which **content creator platform** do you primarily use?
       (e.g., YouTube, Instagram, TikTok, Twitch, etc.)

[State] intent=HIGH_INTENT | stage=lead_collect | lead_email=adarsh@example.com | missing=[platform]

User: YouTube
Agent: 🎉 **Your lead has been captured successfully!**

       **Lead Summary:**
       - **Name:** Adarsh Kumar
       - **Email:** adarsh@example.com
       - **Platform:** YouTube
       - **Plan Interest:** Pro Plan

       Welcome aboard, **Adarsh Kumar**! Our team will reach out to you at
       **adarsh@example.com** within 24 hours.

[State] intent=HIGH_INTENT | stage=captured | is_tool_called=True
[Tool] Lead captured successfully: Adarsh Kumar, adarsh@example.com, YouTube
```

## 5. Post-Capture Follow-up

```
User: Can I upgrade from Basic to Pro later?
Agent: Yes! When you upgrade, the remaining days on your Basic plan are prorated
       and applied as credit toward your Pro subscription.

[State] intent=PRODUCT_QUERY | stage=captured | is_tool_called=True
[RAG] Retrieved chunk: "FAQ — Can I upgrade from Basic to Pro mid-cycle?"
```

## 6. Edge Case: Invalid Email

```
User: I want to subscribe!
Agent: Could you please share your **full name**?

User: Jane
Agent: Could you please provide your **professional email address**?

User: jane-at-gmail
Agent: Could you please provide your **professional email address**?
       (The email format doesn't appear valid — please use format: name@domain.com)

[State] missing=[email, platform] — email extraction returned None
```
